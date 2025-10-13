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

// Matrix-free wrapper: 
//  y_lambda = Bs * ( T_p2s * x_parent )      // via Mult
//  y_parent = T^T * ( Bs^T * y_lambda )  // via MultTranspose
// use on parent space.
class SubmeshOperator : public Operator
{
public:
  SubmeshOperator(const Operator &Bs_, //operator on submesh.
                    FiniteElementSpace &fes_parent,
                    FiniteElementSpace &fes_sub,
                    FiniteElementSpace &fes_lambda)
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
  const Operator &Bs;
  mutable TransferMap T_p2s;     // parent -> submesh
  mutable TransferMap T_s2p;     // submesh -> parent (acts like T^T)
  mutable GridFunction GF1p, GF2s, GF3l, GF4p;
  
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
         solver->SetPrintLevel(0);
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
   bool noSource = true; // no code for the source enabled.
   int order = 1;

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
   real_t deltaT = endTime/nbrTimeSeg;
   real_t Time = 0.0;
//   
// Creates 2D mesh, divided into equal intervals.
//
   Mesh *mesh = new Mesh();
   *mesh = Mesh::MakeCartesian2D(nbrLengthSeg, nbrTimeSeg,
                                  Element::QUADRILATERAL, false, lenght, endTime, false);
   int dim = mesh->Dimension();
   assert(dim==2); //this software works only for dimension 2.
   mesh->PrintInfo();

   int nbrEl = mesh->GetNE();
   assert(nbrEl == nbrLengthSeg * nbrTimeSeg);

   {
      std::ofstream out("out/meshprint.txt");
      if(printMatrix) mesh->Print(out);
   }

   //
   //create the submesh for the vsrs boundary.
   //
   Array<int> *vsrs_bdr_attrs = new Array<int>;
   vsrs_bdr_attrs->Append(4); // Attribute ID=4 for the left boundary.
   SubMesh *vsrsmesh = new SubMesh(SubMesh::CreateFromBoundary(*mesh, *vsrs_bdr_attrs));
   //set BdrAttribute to 0 meaning they are not boundary, necessary for LFVS.
   for (int i = 0; i < vsrsmesh->GetNBE(); i++)
   {
        vsrsmesh->SetBdrAttribute(i, 0);
   }
   vsrsmesh->FinalizeMesh();
   vsrsmesh->PrintInfo();
   assert(vsrsmesh->GetNE() == nbrTimeSeg);

   {
      std::ofstream out("out/vsrsmeshprint.txt");
      if(printMatrix) vsrsmesh->Print(out);
   }
  
   //
   //create the submesh for the t initial boundary.
   //
   Array<int> *t0_bdr_attrs = new Array<int>;
   t0_bdr_attrs->Append(1); // Attribute ID=1 for the bottom boundary.
   SubMesh *t0mesh = new SubMesh(SubMesh::CreateFromBoundary(*mesh, *t0_bdr_attrs));
   //set BdrAttribute to 0 meaning they are not boundary.
   for (int i = 0; i < t0mesh->GetNBE(); i++)
   {
        t0mesh->SetBdrAttribute(i, 0);
   }
   t0mesh->FinalizeMesh();
   t0mesh->PrintInfo();
   assert(t0mesh->GetNE() == nbrLengthSeg);

   {
      std::ofstream out("out/t0meshprint.txt");
      if(printMatrix) t0mesh->Print(out);
   }
  
   //
   // Create the spaces.
   //

   //space for voltage.
   H1_FECollection *VFEC = new H1_FECollection(order, 2);
   FiniteElementSpace *VFESpace = new FiniteElementSpace(mesh, VFEC);
   int VnbrDof = VFESpace->GetTrueVSize(); 
   cout << VnbrDof << " VFESpace degree of freedom\n";  

   //space for current.
   L2_FECollection *IFEC = new L2_FECollection(order, 2, BasisType::GaussLobatto);
   FiniteElementSpace *IFESpace = new FiniteElementSpace(mesh, IFEC);
   int InbrDof = IFESpace->GetTrueVSize(); 
   cout << InbrDof << " IFESpace degree of freedom\n";   
   {
      std::ofstream out("out/ifespace.txt");
      if(printMatrix) IFESpace->Save(out);
   }
   
   //The next three spaces are for application of voltage.
   //space for lagrange multiplier 1 relate to Vs Rs,
   //which cause boundary cnditions on I(0,y).
   
   H1_FECollection *LM1FEC = new H1_FECollection(order, 1);
   FiniteElementSpace *LM1FESpace = new FiniteElementSpace(vsrsmesh, LM1FEC);
   int LM1nbrDof = LM1FESpace->GetTrueVSize(); 
   cout << LM1nbrDof << " LM1FESpace degree of freedom\n";

   //space for voltage over vsrs submesh.
   H1_FECollection *VvsrsFEC = new H1_FECollection(order, 1);
   FiniteElementSpace *VvsrsFESpace = new FiniteElementSpace(vsrsmesh, VvsrsFEC);
   int VvsrsnbrDof = VvsrsFESpace->GetTrueVSize(); 
   cout << VvsrsnbrDof << " VFvsrsESpace degree of freedom\n";   

   //space for current over vsrs submesh. .
   L2_FECollection *IvsrsFEC = new L2_FECollection(order, 1, BasisType::GaussLobatto);
   FiniteElementSpace *IvsrsFESpace = new FiniteElementSpace(vsrsmesh, IvsrsFEC);
   int IvsrsnbrDof = IvsrsFESpace->GetTrueVSize(); 
   cout << IvsrsnbrDof << " IvsrsFESpace degree of freedom\n";   
   {
      std::ofstream out("out/ivsrsfespace.txt");
      if(printMatrix) IvsrsFESpace->Save(out);
   }
   
   
   //the next two spaces are for LM2 applying initial condition V=0.0 at y=0, which is time.
   //space for lagrange multiplier 2 relate to t initial.
   //which cause boundary cnditions of zero on V(x, 0)
      
   H1_FECollection *LM2FEC = new H1_FECollection(order, 1);
   FiniteElementSpace *LM2FESpace = new FiniteElementSpace(t0mesh, LM2FEC);
   int LM2nbrDof = LM2FESpace->GetTrueVSize(); 
   cout << LM2nbrDof << " LM2FESpace degree of freedom\n"; 

   //space for voltage over t0mesh submesh.
   H1_FECollection *Vt0FEC = new H1_FECollection(order, 1);
   FiniteElementSpace *Vt0FESpace = new FiniteElementSpace(t0mesh, Vt0FEC);
   int Vt0nbrDof = Vt0FESpace->GetTrueVSize(); 
   cout << Vt0nbrDof << " Vt0FESpace degree of freedom\n"; 
 
   //the next two spaces are for LM3 applying initial condition I=0.0 at y=0, which is time.
   //space for lagrange multiplier 3 relate to t initial.
   //which cause boundary cnditions of zero on VIx, 0)
   
   H1_FECollection *LM3FEC = new H1_FECollection(order, 1);
   FiniteElementSpace *LM3FESpace = new FiniteElementSpace(t0mesh, LM3FEC);
   int LM3nbrDof = LM3FESpace->GetTrueVSize(); 
   cout << LM3nbrDof << " LM3FESpace degree of freedom\n";   

   //space for current over t0mesh submesh.
   L2_FECollection *It0FEC = new L2_FECollection(order, 1, BasisType::GaussLobatto);
   FiniteElementSpace *It0FESpace = new FiniteElementSpace(t0mesh, It0FEC);
   int It0nbrDof = It0FESpace->GetTrueVSize(); 
   cout << It0nbrDof << " It0FESpace degree of freedom\n";  


//
//Create the forms.
//
   //the forms for the equation with dV/dx.
   //BLF_dvdx implements the x dimension V space derivative,

   ConstantCoefficient one(1.0), Small(1e-4);
   Vector xDir(2); xDir = 0.0; xDir(0) = 1.0;
   VectorConstantCoefficient xDirCoeff(xDir);
   MixedBilinearForm *MBLF_dvdx = new MixedBilinearForm(VFESpace /*trial*/, VFESpace /*test*/);
   MBLF_dvdx->AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(xDirCoeff)); //x direction.
   //MBLF_dvdx->AddDomainIntegrator(new DerivativeIntegrator(one, 0)); //x direction.
   MBLF_dvdx->AddDomainIntegrator(new MixedScalarMassIntegrator(Small));
   MBLF_dvdx->Assemble();
   MBLF_dvdx->Finalize();
   cout << MBLF_dvdx->Height() << " MBLF_dvdx Height()." << endl;
   cout << MBLF_dvdx->Width() << " MBLF_dvdx Width()." << endl;
   assert(MBLF_dvdx->Height() == VnbrDof && MBLF_dvdx->Width() == VnbrDof);

   //MBLF_IV implements the y dimension I derivative which is time and
   //the I.
   ConstantCoefficient CC_R(R);
   Vector vLy(2); vLy = 0.0; vLy(1) = L;
   VectorConstantCoefficient CC_Ly(vLy);
   MixedBilinearForm *MBLF_IV = new MixedBilinearForm(IFESpace /*trial*/, VFESpace /*test*/);
   MBLF_IV->AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(CC_Ly));
   MBLF_IV->AddDomainIntegrator(new MixedScalarMassIntegrator(CC_R));
   MBLF_IV->Assemble();
   MBLF_IV->Finalize();
   cout << MBLF_IV->Height() << " MBLF_IV Height()." << endl;
   cout << MBLF_IV->Width() << " MBLF_IV Width()." << endl;
   assert(MBLF_IV->Height() == VnbrDof);
   assert(MBLF_IV->Width() == InbrDof);
   
   
   //the forms for the equation with dI/dx.
   //MBLF_didx implements the x dimension I space derivative,
   MixedBilinearForm *MBLF_didx = new MixedBilinearForm(IFESpace /*trial*/, IFESpace /*test*/);
   MBLF_didx->AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(xDirCoeff)); //x direction.
   //MBLF_didx->AddDomainIntegrator(new DerivativeIntegrator(one, 0)); //x direction.
   MBLF_didx->AddDomainIntegrator(new MixedScalarMassIntegrator(Small));
   MBLF_didx->Assemble();
   MBLF_didx->Finalize();
   cout << MBLF_didx->Height() << " MBLF_didx Height()." << endl;
   cout << MBLF_didx->Width() << " MBLF_didx Width()." << endl;
   assert(MBLF_didx->Height() == InbrDof && MBLF_didx->Width() == InbrDof );

   //MBLF_VI implements the y dimension V derivative which is time and
   //the V.
   ConstantCoefficient CC_G(G);
   Vector vCy(2); vCy = 0.0; vCy(1) = C;
   VectorConstantCoefficient CC_Cy(vCy);
   MixedBilinearForm *MBLF_VI = new MixedBilinearForm(VFESpace, IFESpace);
   MBLF_VI->AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(CC_Cy));
   MBLF_VI->AddDomainIntegrator(new MixedScalarMassIntegrator(CC_G));
   MBLF_VI->Assemble();
   MBLF_VI->Finalize();
   cout << MBLF_VI->Height() << " MBLF_VI Height()." << endl;
   cout << MBLF_VI->Width() << " MBLF_VI Width()." << endl;
   assert(MBLF_VI->Height() == InbrDof);
   assert(MBLF_VI->Width() == VnbrDof);

   //Mixed Bilinear form VL1.
   MixedBilinearForm *MBLF_VL1 = new MixedBilinearForm(VvsrsFESpace /*trial*/, LM1FESpace /*test*/);
   ConstantCoefficient oneOverRs(1.0/Rs);
   MBLF_VL1->AddDomainIntegrator(new MixedScalarMassIntegrator(oneOverRs));
   MBLF_VL1->Assemble();
   MBLF_VL1->Finalize();
   cout << MBLF_VL1->Height() << " MBLF_VL1 Height()." << endl;
   cout << MBLF_VL1->Width() << " MBLF_VL1 Width()." << endl;
   assert(MBLF_VL1->Height() == LM1nbrDof);
   assert(MBLF_VL1->Width() == VvsrsnbrDof);
   //create the submesh to parent operators for the Lagrange multiplier.
   SubmeshOperator *MBLF_VL1_map = new SubmeshOperator(MBLF_VL1->SpMat(), *VFESpace, *VvsrsFESpace, *LM1FESpace);
   cout << MBLF_VL1_map->Height() << " MBLF_VL1_map Height()." << endl;
   cout << MBLF_VL1_map->Width() << " MBLF_VL1_map Width()." << endl;
   assert(MBLF_VL1_map->Height() == LM1nbrDof);
   assert(MBLF_VL1_map->Width() == VnbrDof);

   //Mixed Bilinear form IL1.
   MixedBilinearForm *MBLF_IL1 = new MixedBilinearForm(IvsrsFESpace /*trial*/, LM1FESpace /*test*/);
   MBLF_IL1->AddDomainIntegrator(new MixedScalarMassIntegrator(one));
   MBLF_IL1->Assemble();
   MBLF_IL1->Finalize();
   cout << MBLF_IL1->Height() << " MBLF_IL1 Height()." << endl;
   cout << MBLF_IL1->Width() << " MBLF_IL1 Width()." << endl;
   assert(MBLF_IL1->Height() == LM1nbrDof);
   assert(MBLF_IL1->Width() == IvsrsnbrDof);
   //create the submesh tp parent operators for the Lagrange multiplier.
   SubmeshOperator *MBLF_IL1_map = new SubmeshOperator(MBLF_IL1->SpMat(), *IFESpace, *IvsrsFESpace, *LM1FESpace);
   cout << MBLF_IL1_map->Height() << " MBLF_IL1_map Height()." << endl;
   cout << MBLF_IL1_map->Width() << " MBLF_IL1_map Width()." << endl;
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
   MixedBilinearForm *MBLF_VL2 = new MixedBilinearForm(Vt0FESpace /*trial*/, LM2FESpace /*test*/);
   MBLF_VL2->AddDomainIntegrator(new MixedScalarMassIntegrator(one));
   MBLF_VL2->Assemble();
   MBLF_VL2->Finalize();
   cout << MBLF_VL2->Height() << " MBLF_VL2 Height()." << endl;
   cout << MBLF_VL2->Width() << " MBLF_VL2 Width()." << endl;
   //create the submesh to parent operators for the Lagrange multiplier.DomainLFIntegrator
   SubmeshOperator *MBLF_VL2_map = new SubmeshOperator(MBLF_VL2->SpMat(), *VFESpace, *Vt0FESpace, *LM2FESpace);

   //Mixed Bilinear form IL3.
   MixedBilinearForm *MBLF_IL3 = new MixedBilinearForm(It0FESpace /*trial*/, LM3FESpace /*test*/);
   MBLF_IL3->AddDomainIntegrator(new MixedScalarMassIntegrator(one));
   MBLF_IL3->Assemble();
   MBLF_IL3->Finalize();
   cout << MBLF_IL3->Height() << " MBLF_IL3 Height()." << endl;
   cout << MBLF_IL3->Width() << " MBLF_IL3 Width()." << endl;
   //create the submesh tp parent operators for the Lagrange multiplier.
   SubmeshOperator *MBLF_IL3_map = new SubmeshOperator(MBLF_IL3->SpMat(), *IFESpace, *It0FESpace, *LM3FESpace);
 
   {
      std::ofstream out("out/MBLF_VL2.txt");
      if(printMatrix) MBLF_VL2->SpMat().PrintMatlab(out); // instead of Print()
   }

   {
      std::ofstream out("out/MBLF_IL3.txt");
      if(printMatrix) MBLF_IL3->SpMat().PrintMatlab(out); // instead of Print()
   }

   LinearForm *LFVS = new LinearForm(LM1FESpace);
   LFVS->AddDomainIntegrator(new DomainLFIntegrator(VsRsFunctionCoeff));
   LFVS->Assemble();

   cout << LFVS->Size() << " LFVS Size()." << endl;
 
   {
      std::ofstream out("out/lfvs.txt");
      if(printMatrix) LFVS->Print(out, 1); // instead of Print()
   }

//
//Define the inner and outer blockopertor.
//


   // 6. Define the PDE inner BlockStructure BA.
      Array<int> *BArowOffset = new Array<int>(3);
      (*BArowOffset)[0]=0;
      (*BArowOffset)[1]=MBLF_dvdx->Height(); 
      (*BArowOffset)[2]=MBLF_VI->Height();
      BArowOffset->PartialSum();
      {
         std::ofstream out("out/BArowOffset.txt");
         BArowOffset->Print(out, 10);
      }

      Array<int> *BAcolOffset = new Array<int>(3);
      (*BAcolOffset)[0]=0;
      (*BAcolOffset)[1]=MBLF_dvdx->Width(); 
      (*BAcolOffset)[2]=MBLF_IV->Width();
      BAcolOffset->PartialSum();
      {
         std::ofstream out("out/BAcolOffset.txt");
         BAcolOffset->Print(out, 10);
      }

      BlockMatrix *BA = new BlockMatrix(*BArowOffset, *BAcolOffset);

      BA->SetBlock(0, 0, &(MBLF_dvdx->SpMat()));
      BA->SetBlock(0, 1, &(MBLF_IV->SpMat()));

      BA->SetBlock(1, 0, &(MBLF_VI->SpMat()));
      BA->SetBlock(1, 1, &(MBLF_didx->SpMat()));

      
      cout << BA->Height() << " BA->Height()" << endl;
      cout << BA->Width() << " BA->Width()" << endl;

      {
      std::ofstream out("out/BA.txt");
      if(printMatrix) BA->PrintMatlab(out); // instead of Print()
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
      
      cout << BB->Height() << " BB->Height()" << endl;
      cout << BB->Width() << " BB->Width()" << endl;

      {
      std::ofstream out("out/BB.txt");
      if(printMatrix) BB->PrintMatlab(out); // instead of Print()
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
      
      cout << OuterBlock->Height() << " OuterBlock->Height()" << endl;
      cout << OuterBlock->Width() << " OuterBlock->Width()" << endl;

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

   //PB00 pb00(BA);
   //assert(pb00.Height() == VnbrDof + InbrDof);
   //assert(pb00.Width() == VnbrDof + InbrDof);

   if(0)
   {
      DSmoother *pb00 = new DSmoother(*(BA->CreateMonolithic()), 0);
   }
   
   SparseMatrix *BAmono = BA->CreateMonolithic();
   Operator *pb00 = new UMFPackSolver(*BAmono);


   
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
      FGMRESSolver solver;
     // solver.SetFlexible(true);
      solver.SetAbsTol(0);
      solver.SetRelTol(1e-6);
      solver.SetMaxIter(5000);
      solver.SetPrintLevel(1);
      solver.SetKDim(50);
      solver.SetOperator(*OuterBlock);
      solver.SetPreconditioner(P);
      solver.Mult(*bBlock, *xBlock);
      cout << solver.GetFinalRelNorm() << " GetFinalRelNorm" << endl;
   }

   if(0)
   {
      MINRESSolver solver;
      solver.SetAbsTol(0.0);
      solver.SetRelTol(1e-8);
      solver.SetMaxIter(5000);
      solver.SetOperator(*OuterBlock);
      //solver.SetPreconditioner(*prec);
      solver.SetPrintLevel(1);
      solver.Mult(*bBlock, *xBlock);
   }

   GridFunction *VGF = new GridFunction(VFESpace, xV, 0);
   Glvis(mesh, VGF, "Voltage", 8, " keys 'mcjaaa'");

   GridFunction *IGF = new GridFunction(IFESpace, xI, 0);
   Glvis(mesh, IGF, "Current", 8, " keys 'mcjaaa'");

   return 1;
}


