//                                stltferk4
//                                
// Compile with: make stltfe2d, need MFEM version 4.8 and GLVIS-4.3.
//
// Sample runs:  ./stltfe2d
//
/*
Description:  stltfe (single transmission line transient finite element
2D) simulate a single transmission line with various source
signals, source impedances and load impedances. It treat time as a space dimension
to make use of MFEM capability.
 
Written by Denis Lachapelle, 2025
*/

#include <mfem.hpp>
#include "mytools.hpp"
#include <iostream>
#include <math.h>
#include <filesystem>

using namespace std;
using namespace mfem;

real_t timeScaling = 1e9;  // make time and space axis the same size.

#include <mfem.hpp>
using namespace mfem;

/*
class BlockDiagAMG : public Solver
{
public:
   BlockDiagAMG(const Array<int> &offsets,
                HypreParMatrix &A00,
                HypreParMatrix &A11)
      : Solver(offsets.Last()),
        offsets_(offsets),
        amg0_(&A00), amg1_(&A11),
        x0_(offsets[1]-offsets[0]),
        x1_(offsets[2]-offsets[1]),
        y0_(offsets[1]-offsets[0]),
        y1_(offsets[2]-offsets[1])
   {
      // Configure each BoomerAMG as you like
      amg0_.SetPrintLevel(0);
      amg0_.SetSystemsOptions(1);  // if A00 is vector-valued SPD; else omit
      amg0_.SetCoarsenType(8);     // HMIS coarsening (example)
      amg0_.SetRelaxType(6);       // Symmetric SOR/Jacobi (example)

      amg1_.SetPrintLevel(0);
      amg1_.SetCoarsenType(8);
      amg1_.SetRelaxType(6);

      // Recommended: amg0_.SetOperator(A00); amg1_.SetOperator(A11);  // done by constructors

      // This preconditioner is square and maps size N -> N
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      // Split input into blocks
      Vector x0(const_cast<double*>(x.GetData()+offsets_[0]), offsets_[1]-offsets_[0]);
      Vector x1(const_cast<double*>(x.GetData()+offsets_[1]), offsets_[2]-offsets_[1]);

      // Apply AMG per block
      amg0_.Mult(x0, y0_);
      amg1_.Mult(x1, y1_);

      // Stitch output
      y.SetSize(offsets_.Last());
      y.Range(offsets_[0], offsets_[1]-offsets_[0]) = y0_;
      y.Range(offsets_[1], offsets_[2]-offsets_[1]) = y1_;
   }

   // Optional: expose SetOperator to rebuild AMG after updates
   void SetOperators(HypreParMatrix &A00, HypreParMatrix &A11)
   {
      amg0_.SetOperator(A00);
      amg1_.SetOperator(A11);
   }

private:
   const Array<int> &offsets_;
   mutable HypreBoomerAMG amg0_, amg1_;
   mutable Vector x0_, x1_, y0_, y1_;
};
*/


// Matrix-free wrapper: 
//  y_lambda = Bs * ( T_p2s * x_parent )      // via Mult
//  y_parent = T^T * ( Bs^T * y_lambda )  // via MultTranspose
// use on parent space.
class SubmeshOperator : public Operator
{
public:
  SubmeshOperator(const HypreParMatrix &Bs_, //operator on submesh.
                    ParFiniteElementSpace &fes_parent,
                    ParFiniteElementSpace &fes_sub,
                    ParFiniteElementSpace &fes_lambda)
  : Operator(fes_lambda.GetTrueVSize(), fes_parent.GetTrueVSize()),
    Bs(Bs_),
    T_p2s(&fes_parent, &fes_sub), // TransferMap parent to submesh.
    T_s2p(&fes_sub, &fes_parent), // TransferMap submesh to parent.
    GF1p(&fes_parent),
    GF2s(&fes_sub),
    GF3l(&fes_lambda),
    GF4p(&fes_parent)
  { }

  virtual void Mult(const Vector &xp, Vector &yl) const override
  {
    GF1p = xp;
    GF2s = 0.0;
    GF3l = 0.0;
    T_p2s.Transfer(GF1p, GF2s);    // xs = T * xp
    Bs.Mult(GF2s, GF3l);           // ys = Bs * xs
    yl = GF3l;
  }

   virtual void MultTranspose(const Vector &yl, Vector &yp) const override
  {
    GF3l = yl;
    GF2s = 0.0;
    GF4p = 0.0;
    Bs.MultTranspose(GF3l, GF2s); // xs = Bs^T * ys
    T_s2p.Transfer(GF2s, GF4p);   // yp = T * xs
    yp = GF4p;
  }

private:
  mutable HypreParMatrix Bs;
  mutable ParTransferMap T_p2s;     // parent -> submesh
  mutable ParTransferMap T_s2p;     // submesh -> parent (acts like T^T)
  mutable ParGridFunction GF1p, GF2s, GF3l, GF4p;
  
};



// PB11 preconditioner block 11.
class PB11 : public Operator
{
   private:
      // BB Block operator of the lagrange multiplier.
      // BAinv estimate of the inverse of BA computed with class PB00.
      BlockOperator &BB;
      Operator &BAinv;
      mutable Vector temp1, temp2;

      TransposeOperator *BBT;
      ProductOperator *AM1BT;
      ProductOperator *BAM1BT;

      GMRESSolver *solver;
         

   public:
      PB11(BlockOperator &BB_, Operator &BAinv_) :
      Operator(BB_.Height(), BB_.Height()),
      BB(BB_), BAinv(BAinv_)
      {
         assert(BB.Width() == BAinv.Width());
         BBT = new TransposeOperator(BB);
         AM1BT = new ProductOperator(&BAinv, BBT, false, false);
         BAM1BT = new ProductOperator(&BB, AM1BT, false, false);

         solver = new GMRESSolver();
         solver->SetAbsTol(0);
         solver->SetRelTol(1e-2);
         solver->SetMaxIter(50);
         solver->SetPrintLevel(1);
         solver->SetKDim(50);
         solver->SetOperator(*BAM1BT);
      }

      virtual void Mult(const Vector &x, Vector &y) const override
      {
         y = 0.0;
         solver->Mult(x, y);
      }
};



void GetDiagFromOperator(const Operator &op, Vector &diag)
{
    int n = op.Height();
    MFEM_VERIFY(n == op.Width(), "must be square");
    diag.SetSize(n);

    Vector x(n), y(n);
    for (int i = 0; i < n; i++)
    {
        x = 0.0; x(i) = 1.0; // unit vector e_i
        op.Mult(x, y);
        diag(i) = y(i);
    }
}

static real_t SourceFunctionGaussianPulse(const Vector x)
{
   /* gaussian pulse of tw wide centered at tc.*/
   real_t t = x(1); //1 for y-axis.
   real_t tw = 20e-9 * timeScaling;
   real_t tc = 20e-9 * timeScaling;
   if(t<2*tc) return 4.0 * t/(2.0*tc)*(1-t/(2.0*tc)) * exp(-pow(((t-tc)/tw), 2.0));
   else return 0.0;
}

real_t SourceFunctionStep(const Vector x)
{
      //step.
      real_t t = x(1); //1 for y-axis.
      real_t tau = 30e-9 * timeScaling;
      return 1.0 - exp(-t/tau);
}

real_t SourceFunctionSine(const Vector x, real_t t)
{
      return sin(2*M_PI*13e6*t);
}

FunctionCoefficient VsRsFunctionCoeff(SourceFunctionGaussianPulse);


int main(int argc, char *argv[])
{

   // default options...
   bool printMatrix = false; //control matrix saving to file.
   int order = 1;

    // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&printMatrix, "-prm", "--printmatrix", "-dnprm", "--donotprintmatrix",
                  "Print of not the matrix.");
                 
   args.Parse();

   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   
//Erase the content of the /out directory.
   system("rm -f out/*");


   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, occ::, RAJA and OpenMP based on command line options.
   Device device("cpu");
   device.Print();
   MemoryType mt = device.GetMemoryType();
   (void)mt;


   double lenght = 100;
   int nbrLengthSeg = 100;
         
   // Constants for the telegrapher’s equation for RG-58, 50 ohm.
   double L = 250e-9 * timeScaling;  // Inductance per unit length
   double C = 100e-12 * timeScaling; // Capacitance per unit length
   double R = 220e-3;  // Resistance per unit length
   double G = 1.0e-7;  // Conductance per unit length

   double Rs = 50.0;
   
   int nbrTimeSeg = 100;
   real_t endTime = 100e-9 * timeScaling;
//   
// Creates 2D mesh, divided into equal intervals.
//
   Mesh *mesh = new Mesh();
   *mesh = Mesh::MakeCartesian2D(nbrLengthSeg, nbrTimeSeg,
                                  Element::QUADRILATERAL, false, lenght, endTime, false);
   int dim = mesh->Dimension();
   assert(dim==2); //this software works only for dimension 2.
   
   if (Mpi::Root())
   {
      mesh->PrintInfo();
   }
   

   // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh once in parallel to increase the resolution.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   

   int nbrEl = mesh->GetNE();
   assert(nbrEl == nbrLengthSeg * nbrTimeSeg);

   if (Mpi::Root() && printMatrix)
   {
      std::ofstream out("out/meshprint.txt");
      mesh->Print(out);
   }

   //
   //create the submesh for the vsrs boundary.
   //
   Array<int> *vsrs_bdr_attrs = new Array<int>;
   vsrs_bdr_attrs->Append(4); // Attribute ID=4 for the left boundary.
   ParSubMesh *vsrsmesh = new ParSubMesh(ParSubMesh::CreateFromBoundary(*pmesh, *vsrs_bdr_attrs));
   //set BdrAttribute to 0 meaning they are not boundary, necessary for LFVS.
   //for (int i = 0; i < vsrsmesh->GetNBE(); i++)
   //{
   //     vsrsmesh->SetBdrAttribute(i, 0);
   //}
   vsrsmesh->FinalizeMesh(0, true);
   if (Mpi::Root()) vsrsmesh->PrintInfo();
   assert(vsrsmesh->GetNE() == nbrTimeSeg);

   if (Mpi::Root() && printMatrix)
   {
      std::ofstream out("out/vsrsmeshprint.txt");
      vsrsmesh->Print(out);
   }
  
   //
   //create the submesh for the t initial boundary.
   //
   Array<int> *t0_bdr_attrs = new Array<int>;
   t0_bdr_attrs->Append(1); // Attribute ID=1 for the bottom boundary.
   ParSubMesh *t0mesh = new ParSubMesh(ParSubMesh::CreateFromBoundary(*pmesh, *t0_bdr_attrs));
   //set BdrAttribute to 0 meaning they are not boundary.
   //for (int i = 0; i < t0mesh->GetNBE(); i++)
   //{
   //    t0mesh->SetBdrAttribute(i, 0);
   //}
   t0mesh->FinalizeMesh();
   if (Mpi::Root()) t0mesh->PrintInfo();
   assert(t0mesh->GetNE() == nbrLengthSeg);

   if (Mpi::Root() && printMatrix)
   {
      std::ofstream out("out/t0meshprint.txt");
      t0mesh->Print(out);
   }
  
   //
   // Create the spaces.
   //

   //space for voltage.
   H1_FECollection *VFEC = new H1_FECollection(order, 2);
   ParFiniteElementSpace *VFESpace = new ParFiniteElementSpace(pmesh, VFEC);
   int VnbrDof = VFESpace->GetTrueVSize(); 
   if (Mpi::Root()) cout << VnbrDof << " VFESpace degree of freedom\n";  

   

   //space for current.
   H1_FECollection *IFEC = new H1_FECollection(order, 2);
   ParFiniteElementSpace *IFESpace = new ParFiniteElementSpace(pmesh, IFEC);
   int InbrDof = IFESpace->GetTrueVSize(); 
   if (Mpi::Root()) cout << InbrDof << " IFESpace degree of freedom\n";   
   if (Mpi::Root() && printMatrix)
   {
      std::ofstream out("out/ifespace.txt");
      IFESpace->Save(out);
   }


   //The next three spaces are for application of voltage.
   //space for lagrange multiplier 1 relate to Vs Rs,
   //which cause boundary cnditions on I(0,y).
   
   H1_FECollection *LM1FEC = new H1_FECollection(order, 1);
   ParFiniteElementSpace *LM1FESpace = new ParFiniteElementSpace(vsrsmesh, LM1FEC);
   int LM1nbrDof = LM1FESpace->GetTrueVSize(); 
   if (Mpi::Root()) cout << LM1nbrDof << " LM1FESpace degree of freedom\n";

   //space for voltage over vsrs submesh.
   H1_FECollection *VvsrsFEC = new H1_FECollection(order, 1);
   ParFiniteElementSpace *VvsrsFESpace = new ParFiniteElementSpace(vsrsmesh, VvsrsFEC);
   int VvsrsnbrDof = VvsrsFESpace->GetTrueVSize(); 
   if (Mpi::Root()) cout << VvsrsnbrDof << " VFvsrsESpace degree of freedom\n";   

   //space for current over vsrs submesh. .
   H1_FECollection *IvsrsFEC = new H1_FECollection(order, 1);
   ParFiniteElementSpace *IvsrsFESpace = new ParFiniteElementSpace(vsrsmesh, IvsrsFEC);
   int IvsrsnbrDof = IvsrsFESpace->GetTrueVSize(); 
   if (Mpi::Root()) cout << IvsrsnbrDof << " IvsrsFESpace degree of freedom\n";   
   {
      std::ofstream out("out/ivsrsfespace.txt");
      if(printMatrix) IvsrsFESpace->Save(out);
   }
   
   
   //the next two spaces are for LM2 applying initial condition V=0.0 at y=0, which is time.
   //space for lagrange multiplier 2 relate to t initial.
   //which cause boundary cnditions of zero on V(x, 0)
      
   H1_FECollection *LM2FEC = new H1_FECollection(order, 1);
   ParFiniteElementSpace *LM2FESpace = new ParFiniteElementSpace(t0mesh, LM2FEC);
   int LM2nbrDof = LM2FESpace->GetTrueVSize(); 
   if (Mpi::Root()) cout << LM2nbrDof << " LM2FESpace degree of freedom\n"; 

   //space for voltage over t0mesh submesh.
   H1_FECollection *Vt0FEC = new H1_FECollection(order, 1);
   ParFiniteElementSpace *Vt0FESpace = new ParFiniteElementSpace(t0mesh, Vt0FEC);
   int Vt0nbrDof = Vt0FESpace->GetTrueVSize(); 
   if (Mpi::Root()) cout << Vt0nbrDof << " Vt0FESpace degree of freedom\n"; 
 
   //the next two spaces are for LM3 applying initial condition I=0.0 at y=0, which is time.
   //space for lagrange multiplier 3 relate to t initial.
   //which cause boundary cnditions of zero on VIx, 0)
   
   H1_FECollection *LM3FEC = new H1_FECollection(order, 1);
   ParFiniteElementSpace *LM3FESpace = new ParFiniteElementSpace(t0mesh, LM3FEC);
   int LM3nbrDof = LM3FESpace->GetTrueVSize(); 
   if (Mpi::Root()) cout << LM3nbrDof << " LM3FESpace degree of freedom\n";   

   //space for current over t0mesh submesh.
   H1_FECollection *It0FEC = new H1_FECollection(order, 1, BasisType::GaussLobatto);
   ParFiniteElementSpace *It0FESpace = new ParFiniteElementSpace(t0mesh, It0FEC);
   int It0nbrDof = It0FESpace->GetTrueVSize(); 
   if (Mpi::Root()) cout << It0nbrDof << " It0FESpace degree of freedom\n";  


//
//Create the forms.
//
   //the forms for the equation with dV/dx.
   //BLF_dvdx implements the x dimension V space derivative,

   ConstantCoefficient one(1.0), Small(1e-4);
   Vector xDir(2); xDir = 0.0; xDir(0) = 1.0;
   VectorConstantCoefficient xDirCoeff(xDir);
   ParMixedBilinearForm *MBLF_dvdx = new ParMixedBilinearForm(VFESpace /*trial*/, VFESpace /*test*/);
   MBLF_dvdx->AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(xDirCoeff)); //x direction.
   //MBLF_dvdx->AddDomainIntegrator(new DerivativeIntegrator(one, 0)); //x direction.
   MBLF_dvdx->AddDomainIntegrator(new MixedScalarMassIntegrator(Small));
   MBLF_dvdx->Assemble();
   MBLF_dvdx->Finalize();
   HypreParMatrix *pMBLF_dvdx = MBLF_dvdx->ParallelAssemble();
   if (Mpi::Root())  cout  << MBLF_dvdx->Height() << " MBLF_dvdx Height()." << endl;
   if (Mpi::Root())  cout  << MBLF_dvdx->Width() << " MBLF_dvdx Width()." << endl;
   assert(MBLF_dvdx->Height() == VnbrDof && MBLF_dvdx->Width() == VnbrDof);

   //MBLF_IV implements the y dimension I derivative which is time and
   //the I.
   ConstantCoefficient CC_R(R);
   Vector vLy(2); vLy = 0.0; vLy(1) = L;
   VectorConstantCoefficient CC_Ly(vLy);
   ParMixedBilinearForm *MBLF_IV = new ParMixedBilinearForm(IFESpace /*trial*/, VFESpace /*test*/);
   MBLF_IV->AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(CC_Ly));
   MBLF_IV->AddDomainIntegrator(new MixedScalarMassIntegrator(CC_R));
   MBLF_IV->Assemble();
   MBLF_IV->Finalize();
   HypreParMatrix *pMBLF_IV = MBLF_IV->ParallelAssemble();
   if (Mpi::Root())  cout  << MBLF_IV->Height() << " MBLF_IV Height()." << endl;
   if (Mpi::Root())  cout  << MBLF_IV->Width() << " MBLF_IV Width()." << endl;
   assert(MBLF_IV->Height() == VnbrDof);
   assert(MBLF_IV->Width() == InbrDof);
   
   
   //the forms for the equation with dI/dx.
   //MBLF_didx implements the x dimension I space derivative,
   ParMixedBilinearForm *MBLF_didx = new ParMixedBilinearForm(IFESpace /*trial*/, IFESpace /*test*/);
   MBLF_didx->AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(xDirCoeff)); //x direction.
   //MBLF_didx->AddDomainIntegrator(new DerivativeIntegrator(one, 0)); //x direction.
   MBLF_didx->AddDomainIntegrator(new MixedScalarMassIntegrator(Small));
   MBLF_didx->Assemble();
   MBLF_didx->Finalize();
   HypreParMatrix *pMBLF_didx = MBLF_didx->ParallelAssemble();
   if (Mpi::Root())  cout  << MBLF_didx->Height() << " MBLF_didx Height()." << endl;
   if (Mpi::Root())  cout  << MBLF_didx->Width() << " MBLF_didx Width()." << endl;
   assert(MBLF_didx->Height() == InbrDof && MBLF_didx->Width() == InbrDof );

   //MBLF_VI implements the y dimension V derivative which is time and
   //the V.
   ConstantCoefficient CC_G(G);
   Vector vCy(2); vCy = 0.0; vCy(1) = C;
   VectorConstantCoefficient CC_Cy(vCy);
   ParMixedBilinearForm *MBLF_VI = new ParMixedBilinearForm(VFESpace, IFESpace);
   MBLF_VI->AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(CC_Cy));
   MBLF_VI->AddDomainIntegrator(new MixedScalarMassIntegrator(CC_G));
   MBLF_VI->Assemble();
   MBLF_VI->Finalize();
   HypreParMatrix *pMBLF_VI = MBLF_VI->ParallelAssemble();
   if (Mpi::Root())  cout  << MBLF_VI->Height() << " MBLF_VI Height()." << endl;
   if (Mpi::Root())  cout  << MBLF_VI->Width() << " MBLF_VI Width()." << endl;
   assert(MBLF_VI->Height() == InbrDof);
   assert(MBLF_VI->Width() == VnbrDof);

   //Mixed Bilinear form VL1.
   ParMixedBilinearForm *MBLF_VL1 = new ParMixedBilinearForm(VvsrsFESpace /*trial*/, LM1FESpace /*test*/);
   ConstantCoefficient oneOverRs(1.0/Rs);
   MBLF_VL1->AddDomainIntegrator(new MixedScalarMassIntegrator(oneOverRs));
   MBLF_VL1->Assemble();
   MBLF_VL1->Finalize();
   HypreParMatrix *pMBLF_VL1 = MBLF_VL1->ParallelAssemble();
   if (Mpi::Root())  cout  << MBLF_VL1->Height() << " MBLF_VL1 Height()." << endl;
   if (Mpi::Root())  cout  << MBLF_VL1->Width() << " MBLF_VL1 Width()." << endl;
   assert(MBLF_VL1->Height() == LM1nbrDof);
   assert(MBLF_VL1->Width() == VvsrsnbrDof);
   //create the submesh to parent operators for the Lagrange multiplier.
   SubmeshOperator *MBLF_VL1_map = new SubmeshOperator(*pMBLF_VL1, *VFESpace, *VvsrsFESpace, *LM1FESpace);
   if (Mpi::Root())  cout  << MBLF_VL1_map->Height() << " MBLF_VL1_map Height()." << endl;
   if (Mpi::Root())  cout  << MBLF_VL1_map->Width() << " MBLF_VL1_map Width()." << endl;
   assert(MBLF_VL1_map->Height() == LM1nbrDof);
   assert(MBLF_VL1_map->Width() == VnbrDof);

   //Mixed Bilinear form IL1.
   ParMixedBilinearForm *MBLF_IL1 = new ParMixedBilinearForm(IvsrsFESpace /*trial*/, LM1FESpace /*test*/);
   MBLF_IL1->AddDomainIntegrator(new MixedScalarMassIntegrator(one));
   MBLF_IL1->Assemble();
   MBLF_IL1->Finalize();
   HypreParMatrix *pMBLF_IL1 = MBLF_IL1->ParallelAssemble();
   if (Mpi::Root())  cout  << MBLF_IL1->Height() << " MBLF_IL1 Height()." << endl;
   if (Mpi::Root())  cout  << MBLF_IL1->Width() << " MBLF_IL1 Width()." << endl;
   assert(MBLF_IL1->Height() == LM1nbrDof);
   assert(MBLF_IL1->Width() == IvsrsnbrDof);
   //create the submesh tp parent operators for the Lagrange multiplier.
   SubmeshOperator *MBLF_IL1_map = new SubmeshOperator(*pMBLF_IL1, *IFESpace, *IvsrsFESpace, *LM1FESpace);
   if (Mpi::Root())  cout  << MBLF_IL1_map->Height() << " MBLF_IL1_map Height()." << endl;
   if (Mpi::Root())  cout  << MBLF_IL1_map->Width() << " MBLF_IL1_map Width()." << endl;
   assert(MBLF_IL1_map->Height() == LM1nbrDof);
   assert(MBLF_IL1_map->Width() == InbrDof);

   {
      std::ofstream out("out/MBLF_VL1.txt");
      if(printMatrix) MBLF_VL1->SpMat().PrintMatlab(out); // instead of Print()
   }

   {
      std::ofstream out("out/MBLF_IL1.txt");
      if(printMatrix) MBLF_IL1->SpMat().PrintMatlab(out); // instead of Print()
   }
   
   //Mixed Bilinear form VL2.
   ParMixedBilinearForm *MBLF_VL2 = new ParMixedBilinearForm(Vt0FESpace /*trial*/, LM2FESpace /*test*/);
   MBLF_VL2->AddDomainIntegrator(new MixedScalarMassIntegrator(one));
   MBLF_VL2->Assemble();
   MBLF_VL2->Finalize();
   HypreParMatrix *pMBLF_VL2 = MBLF_VL2->ParallelAssemble();
   if (Mpi::Root())  cout  << MBLF_VL2->Height() << " MBLF_VL2 Height()." << endl;
   if (Mpi::Root())  cout  << MBLF_VL2->Width() << " MBLF_VL2 Width()." << endl;
   //create the submesh to parent operators for the Lagrange multiplier.DomainLFIntegrator
   SubmeshOperator *MBLF_VL2_map = new SubmeshOperator(*pMBLF_VL2, *VFESpace, *Vt0FESpace, *LM2FESpace);

   //Mixed Bilinear form IL3.
   ParMixedBilinearForm *MBLF_IL3 = new ParMixedBilinearForm(It0FESpace /*trial*/, LM3FESpace /*test*/);
   MBLF_IL3->AddDomainIntegrator(new MixedScalarMassIntegrator(one));
   MBLF_IL3->Assemble();
   MBLF_IL3->Finalize();
   HypreParMatrix *pMBLF_IL3 = MBLF_IL3->ParallelAssemble();
   if (Mpi::Root())  cout  << MBLF_IL3->Height() << " MBLF_IL3 Height()." << endl;
   if (Mpi::Root())  cout  << MBLF_IL3->Width() << " MBLF_IL3 Width()." << endl;
   //create the submesh tp parent operators for the Lagrange multiplier.
   SubmeshOperator *MBLF_IL3_map = new SubmeshOperator(*pMBLF_IL3, *IFESpace, *It0FESpace, *LM3FESpace);
 
   {
      std::ofstream out("out/MBLF_VL2.txt");
      if(printMatrix) MBLF_VL2->SpMat().PrintMatlab(out); // instead of Print()
   }

   {
      std::ofstream out("out/MBLF_IL3.txt");
      if(printMatrix) MBLF_IL3->SpMat().PrintMatlab(out); // instead of Print()
   }

   ParLinearForm *LFVS = new ParLinearForm(LM1FESpace);
   LFVS->AddDomainIntegrator(new DomainLFIntegrator(VsRsFunctionCoeff));
   LFVS->Assemble();

   if (Mpi::Root())  cout  << LFVS->Size() << " LFVS Size()." << endl;
 
   {
      std::ofstream out("out/lfvs.txt");
      if(printMatrix) LFVS->Print(out, 1); // instead of Print()
   }

//
//Define the inner and outer blockopertor.
//


   // 6. Define the PDE inner BlockStructure BA.
      Array<int> *BAOffset = new Array<int>(3);
      (*BAOffset)[0]=0;
      (*BAOffset)[1]=pMBLF_dvdx->Height(); 
      (*BAOffset)[2]=pMBLF_VI->Height();
      BAOffset->PartialSum();
      {
         std::ofstream out("out/BArowOffset.txt");
         BAOffset->Print(out, 10);
      }

      BlockOperator *BA = new BlockOperator(*BAOffset);

      BA->SetBlock(0, 0, pMBLF_dvdx);
      BA->SetBlock(0, 1, pMBLF_IV);

      BA->SetBlock(1, 0, pMBLF_VI);
      BA->SetBlock(1, 1, pMBLF_didx);

      
      if (Mpi::Root())  cout  << BA->Height() << " BA->Height()" << endl;
      if (Mpi::Root())  cout  << BA->Width() << " BA->Width()" << endl;

      if (Mpi::Root() && printMatrix)
      {
      std::ofstream out("out/BA.txt");
      BA->PrintMatlab(out); // instead of Print()
      }

      // 6. Define the constraints inner BlockStructure BB.
      Array<int> *BBrowOffset = new Array<int>(4);
      (*BBrowOffset)[0]=0;
      (*BBrowOffset)[1]=MBLF_VL1_map->Height(); 
      (*BBrowOffset)[2]=MBLF_VL2_map->Height();
      (*BBrowOffset)[3]=MBLF_IL3_map->Height();
      BBrowOffset->PartialSum();
      
      {
         std::ofstream out("out/BBrowOffset.txt");
         BBrowOffset->Print(out, 10);
      }

      Array<int> *BBcolOffset = new Array<int>(3);
      (*BBcolOffset)[0]=0;
      (*BBcolOffset)[1]=MBLF_VL1_map->Width(); 
      (*BBcolOffset)[2]=MBLF_IL1_map->Width();
      BBcolOffset->PartialSum();
      
      {
         std::ofstream out("out/BBcolOffset.txt");
         BBcolOffset->Print(out, 10);
      }

      BlockOperator *BB = new BlockOperator(*BBrowOffset, *BBcolOffset);

      BB->SetBlock(0, 0, MBLF_VL1_map);
      BB->SetBlock(0, 1, MBLF_IL1_map);
      BB->SetBlock(1, 0, MBLF_VL2_map);
      BB->SetBlock(2, 1, MBLF_IL3_map);
      
      if (Mpi::Root())  cout  << BB->Height() << " BB->Height()" << endl;
      if (Mpi::Root())  cout  << BB->Width() << " BB->Width()" << endl;

      if (Mpi::Root() && printMatrix) {
         std::ofstream out("out/BB.txt");
         BB->PrintMatlab(out); // instead of Print()
      }

      //
      // create the outer block structure composed of the inner blocks.
      //
           
      
      Array<int> *OuterBlockOffsets = new Array<int>(3);
      (*OuterBlockOffsets)[0]=0;
      (*OuterBlockOffsets)[1]=BA->Height(); 
      (*OuterBlockOffsets)[2]=BB->Height();
      OuterBlockOffsets->PartialSum();
      
      {
         std::ofstream out("out/OuterBlockOffsets.txt");
         OuterBlockOffsets->Print(out, 10);
      }

      BlockOperator *OuterBlock = new BlockOperator(*OuterBlockOffsets);

      TransposeOperator *BBT = new TransposeOperator(BB);

      OuterBlock->SetBlock(0, 0, BA);
      OuterBlock->SetBlock(0, 1, BBT);
      OuterBlock->SetBlock(1, 0, BB);
      
      if (Mpi::Root())  cout  << OuterBlock->Height() << " OuterBlock->Height()" << endl;
      if (Mpi::Root())  cout  << OuterBlock->Width() << " OuterBlock->Width()" << endl;

      {
      std::ofstream out("out/OuterBlock.txt");
      if(printMatrix) OuterBlock->PrintMatlab(out); // instead of Print()
      }

   //
   //create the x and b vectors.
   //   

      Array<int> VectorOffsets(6);
      VectorOffsets[0] = 0;
      VectorOffsets[1] = MBLF_dvdx->Height();
      VectorOffsets[2] = MBLF_VI->Height();
      VectorOffsets[3] = MBLF_VL1->Height();
      VectorOffsets[4] = MBLF_VL2->Height();
      VectorOffsets[5] = MBLF_IL3->Height();
      VectorOffsets.PartialSum();

      // Create vector x.
      BlockVector *xBlock = new BlockVector(VectorOffsets);
      *xBlock = 0.0;
      // Assign xV and xI to refer to V and I later.
      Vector &xV = xBlock->GetBlock(0);
      Vector &xI = xBlock->GetBlock(1);
      if(printMatrix)
      {
         std::ofstream out("out/xblock.txt");
         xBlock->Print(out, 1);
      }

      // Create vector b.
      BlockVector *bBlock = new BlockVector(VectorOffsets);
      *bBlock = 0.0;
      bBlock->GetBlock(2) = *LFVS;

     
      {
      std::ofstream out("out/bblock.txt");
      if(printMatrix) bBlock->Print(out, 1);
      }

//
// Prepare the preconditionner...
//

   HypreBoomerAMG *BA00 = new HypreBoomerAMG(*pMBLF_dvdx);
   HypreBoomerAMG *BA11 = new HypreBoomerAMG(*pMBLF_didx); 
   BlockDiagonalPreconditioner *pb00 = new BlockDiagonalPreconditioner(*BAOffset);
   pb00->SetDiagonalBlock(0, BA00);
   pb00->SetDiagonalBlock(1, BA11);

   
   PB11 pb11(*BB, *pb00);
   assert(pb11.Height() == LM1nbrDof + LM2nbrDof + LM3nbrDof);
   assert(pb11.Width() == LM1nbrDof + LM2nbrDof + LM3nbrDof);

   //Prec prec(pb00, pb11);

   BlockDiagonalPreconditioner P(*OuterBlockOffsets);
   P.SetDiagonalBlock(0, pb00);
   P.SetDiagonalBlock(1, &pb11);


//
// Solve the equations system.
//
   if(1)
   {
      FGMRESSolver solver(MPI_COMM_WORLD);
     // solver.SetFlexible(true);
      solver.SetAbsTol(0);
      solver.SetRelTol(1e-6);
      solver.SetMaxIter(5000);
      solver.SetPrintLevel(1);
      solver.SetKDim(50);
      solver.SetOperator(*OuterBlock);
      solver.SetPreconditioner(P);
      solver.Mult(*bBlock, *xBlock);
      if (Mpi::Root())  cout  << solver.GetFinalRelNorm() << " GetFinalRelNorm" << endl;
   }

  

   ParGridFunction *VGF = new ParGridFunction(VFESpace, xV, 0);
   pGlvis(myid, pmesh, VGF, "Voltage", 8, " keys 'mcjaaa'");

   ParGridFunction *IGF = new ParGridFunction(IFESpace, xI, 0);
   pGlvis(myid, pmesh, IGF, "Current", 8, " keys 'mcjaaa'");

   return 1;
}


