//                                stltfe2d_submesh
//                                
// Compile with: make stltfe2d_submesh, need MFEM version 4.8 and GLVIS-4.3.
//
// Sample runs:  ./stltfe2d_submesh
//
/*
Description:  stltfe_submesh(single transmission line transient finite element
2D) simulate a single transmission line with various source
signals, source impedances and load impedances. It treat time as a space dimension (y)
to make use of MFEM capability.
 
Written by Denis Lachapelle, 2025
restart to work on it Dec 1, 2025. (This software never work.)
*/

#include <mfem.hpp>
#include "mytools.hpp"
#include <iostream>
#include <math.h>
#include <filesystem>

using namespace std;
using namespace mfem;

real_t lenght = 1;                     // transmission line lenght.
real_t RawEndTime = 1;              // Unscale End time.
real_t timeScaling = lenght/RawEndTime;  // make time and space axis the same size (space size).
    
int nbrLengthSeg = 200;
int nbrTimeSeg = 200;
real_t endTime = RawEndTime * timeScaling;
real_t deltaT = endTime/nbrTimeSeg;
real_t Time = 0.0;

real_t Rs = 50.0;

   // Constants for the telegrapher’s equation for RG-58, 50 ohm.
real_t L = 1 * timeScaling;  // Inductance per unit length
real_t C = 1 * timeScaling; // Capacitance per unit length
real_t R = 0.1;  // Resistance per unit length
real_t G = 1e-6;  // Conductance per unit length

// Function to build the LinearForm rhs.
real_t IL1Func(const Vector vec)
{
   real_t x = vec[0], t = vec[1];
   real_t out;
   if(x == 0.0) out = Rs;
   else if(t == 0.0) out = 0.0;
   else MFEM_VERIFY(false, "error in ""real_t IL1Func(const Vector vec)""")
   return out;
}


// Matrix-free wrapper: t options...
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

/* This class implement an operator applying the inverse of Schur complement.
   Ainv_: Operator applying the aproximative inverse of the system matrix.
   B_: Operator applying Lagrange multiplier matrix.
 */   
class InvSchurComp : public Solver   
{
    private:
       Solver &Ainv;
       SubmeshOperator &B;
       TransposeOperator *BT;
       GMRESSolver *solver;
       ProductOperator *AM1BT, *BAM1BT;
       public:
       InvSchurComp(Solver &Ainv_, SubmeshOperator &B_)
       : Solver(B_.Height()), Ainv(Ainv_), B(B_)
       {

         BT = new TransposeOperator(B);
         AM1BT = new ProductOperator(&Ainv, BT, false, false);
         BAM1BT = new ProductOperator(&B, AM1BT, false, false);

         solver = new GMRESSolver();
         solver->SetAbsTol(0);
         solver->SetRelTol(1e-2);
         solver->SetMaxIter(200);
         solver->SetPrintLevel(1);
         solver->SetKDim(20);
         solver->SetOperator(*BAM1BT);

       }

         virtual void Mult(const Vector &b, Vector &x) const override
         {
            x = 0.0;
            solver->Mult(b, x);
         }

         virtual void SetOperator(const Operator &op)
         {

         }
};

class CompleteSchurPreconditioner : public Solver
{
      private:
         
         BlockMatrix   *SysInnBlockA;
         BlockOperator *SysInnBlockB;
         Operator *SysInnBlockBT; 
         
         ProductOperator *AM1BT, *BAM1BT;
         
         BlockOperator *outerPrec;
         BlockOperator *innerA_Prec;
         
         DSmoother *A00inv, *A11inv;

         GMRESSolver *solver;

      public:
         CompleteSchurPreconditioner(BlockOperator &blockOp)
         : Solver(blockOp.Height())
         {
            //Fouterblock
            // prepare preconditioner block A
            //                  
            SysInnBlockA = NULL;
            SysInnBlockA = dynamic_cast<BlockMatrix*>(&blockOp.GetBlock(0,0));
            MFEM_VERIFY(SysInnBlockA != NULL, "Expected BlockMatrix in OuterBlock(0,0)");

            outerPrec = new BlockOperator(blockOp.RowOffsets());
            innerA_Prec = new BlockOperator(SysInnBlockA->RowOffsets(), SysInnBlockA->ColOffsets());
            {
               std::ofstream out("out/precrowoffsets.txt");
               SysInnBlockA->RowOffsets().Print(out, 10);
            }
            {
               std::ofstream out("out/preccoloffsets.txt");
               SysInnBlockA->ColOffsets().Print(out, 10);
            }

            A00inv = new DSmoother(2, 1.0, 10);
            A00inv->SetOperator(SysInnBlockA->GetBlock(0, 0));
            A11inv = new DSmoother (2, 1.0, 10);
            A11inv->SetOperator(SysInnBlockA->GetBlock(1, 1));

            innerA_Prec->SetBlock(0, 0, A00inv);
            innerA_Prec->SetBlock(1, 1, A11inv);

            //
            // prepare preconditioner block B
            //
            SysInnBlockB = NULL;
            SysInnBlockB = dynamic_cast<BlockOperator*>(&(blockOp.GetBlock(1, 0)));
            MFEM_VERIFY(SysInnBlockB != NULL, "FAILS: &dynamic_cast<BlockOperator&>(blockOp.GetBlock(1, 0))");

            SysInnBlockBT = NULL;
            SysInnBlockBT = &(blockOp.GetBlock(0, 1));
            MFEM_VERIFY(SysInnBlockBT != NULL, "FAILS: blockOp.GetBlock(0, 1)");
            
            cout << innerA_Prec->Height() << ": innerA_Prec->Height()" << endl;
            cout << innerA_Prec->Width() << ": innerA_Prec->Width()" << endl;
            cout << SysInnBlockBT->Height() << ": SysInnBlockBT->Height()" << endl;
            cout << SysInnBlockBT->Width() << ": SysInnBlockBT->Width()" << endl;            

            AM1BT = new ProductOperator(innerA_Prec, SysInnBlockBT, false, false);
            BAM1BT = new ProductOperator(SysInnBlockB, AM1BT, false, false);

            solver = new GMRESSolver();
            solver->SetAbsTol(0);
            solver->SetRelTol(1e-1);
            solver->SetMaxIter(10);
            solver->SetPrintLevel(0);
            solver->SetKDim(20);
            solver->SetOperator(*BAM1BT);
            
            outerPrec->SetBlock(0, 0, innerA_Prec);
            outerPrec->SetBlock(1, 1, solver);
         }

         virtual void Mult(const Vector &x, Vector &y) const override
         {
            MFEM_ASSERT(x.Size() == Size(), "Preconditioner input size mismatch.");
            MFEM_ASSERT(y.Size() == Size(), "Preconditioner output size mismatch.");

            outerPrec->Mult(x, y);
         }

         virtual void SetOperator(const Operator &op)
         {
            MFEM_ABORT("CompleteSchurPreconditioner::SetOperator not implemented.");
         }


};


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
      real_t tau = 0.1 * timeScaling;
      real_t td = 0.1 *timeScaling;
      real_t out;
      if(t-td >= 0) out = 1.0 - exp(-(t-td)/tau);
      else out = 0.0;
      return out;
}

real_t SourceFunctionSine(const Vector x, real_t t)
{
      return sin(2*M_PI*13e6*t);
}

real_t SourceFunction(const Vector x, real_t t)
{
   real_t tt = x[1];
   real_t xx = x[0];
   real_t out;
   if(tt == 0.0) out = 0.0;
   else if(xx == 0.0) out = SourceFunctionStep(x);
   else MFEM_VERIFY(false, "problem in ""real_t SourceFunction(const Vector x, real_t t)""");   
   return out;
}

FunctionCoefficient VsRsFunctionCoeff(SourceFunction);

int main(int argc, char *argv[])
{
   // default options...
   bool printMatrix = false; //control matrix saving to file.
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

//   
// Creates 2D mesh, divided into equal intervals.
//
   Mesh *mesh = new Mesh();
   *mesh = Mesh::MakeCartesian2D(nbrLengthSeg, nbrTimeSeg,
                                  Element::QUADRILATERAL, false, lenght, endTime, false);
   int dim = mesh->Dimension();
   assert(dim==2); //this software works only for dimension 2age over t0mesh submesh.
   mesh->PrintInfo();
   int nbrEl = mesh->GetNE();
   assert(nbrEl == nbrLengthSeg * nbrTimeSeg);

   {
      std::ofstream out("out/meshprint.txt");
       mesh->Print(out);
   }

   //
   //create the submesh for the vsrs boundary.
   //
   Array<int> *vsrs_bdr_attrs = new Array<int>;
   vsrs_bdr_attrs->Append(1); // Attribute ID=1 for the bottom boundary.
   vsrs_bdr_attrs->Append(4); // Attribute ID=4 for the left boundary.
   SubMesh *vsrsmesh = new SubMesh(SubMesh::CreateFromBoundary(*mesh, *vsrs_bdr_attrs));
   //set BdrAttribute to 0 meaning they are not boundary, necessary for LFVS.
   for (int i = 0; i < vsrsmesh->GetNBE(); i++)
   {
        vsrsmesh->SetBdrAttribute(i, 0);
   }
   vsrsmesh->FinalizeMesh();
   vsrsmesh->PrintInfo();
   assert(vsrsmesh->GetNE() == nbrTimeSeg + nbrLengthSeg);

   {
      std::ofstream out("out/vsrsmeshprint.txt");
      vsrsmesh->Print(out);
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
      t0mesh->Print(out);
   }
   
   //
   // Create the spaces.
   //

   //space for voltage.
   H1_FECollection *VFEC = new H1_FECollection(order, 2);
   FiniteElementSpace *VFESpace = new FiniteElementSpace(mesh, VFEC);
   int VnbrDof = VFESpace->GetTrueVSize(); 
   cout << VnbrDof << " VFESpace degree of freedom\n";  

   {
      std::ofstream out("out/vfespace.txt");
      VFESpace->Save(out);
   }
   
   //space for current.
   H1_FECollection *IFEC = new H1_FECollection(order, 2);
   FiniteElementSpace *IFESpace = new FiniteElementSpace(mesh, IFEC);
   int InbrDof = IFESpace->GetTrueVSize(); 
   cout << InbrDof << " IFESpace degree of freedom\n";   

   {
      std::ofstream out("out/ifespace.txt");
      IFESpace->Save(out);
   }

   //space for lagrange multiplier 1 (LM1).
   H1_FECollection *LM1FEC = new H1_FECollection(order, 1);
   FiniteElementSpace *LM1FESpace = new FiniteElementSpace(vsrsmesh, LM1FEC);
   int LM1nbrDof = LM1FESpace->GetTrueVSize(); 
   cout << LM1nbrDof << " LM1FESpace degree of freedom\n";
   {
      std::ofstream out("out/lm1fespace.txt");
      LM1FESpace->Save(out);
   }
   //space for lagrange multiplier 2 (LM2).
   H1_FECollection *LM2FEC = new H1_FECollection(order, 1);
   FiniteElementSpace *LM2FESpace = new FiniteElementSpace(t0mesh, LM2FEC);
   int LM2nbrDof = LM2FESpace->GetTrueVSize(); 
   cout << LM2nbrDof << " LM2FESpace degree of freedom\n"; 
   {
      std::ofstream out("out/lm2fespace.txt");
      LM2FESpace->Save(out);
   }
   //sub space for voltage over vsrs submesh.
   H1_FECollection *VvsrsFEC = new H1_FECollection(order, 1);
   FiniteElementSpace *VvsrsFESpace = new FiniteElementSpace(vsrsmesh, VvsrsFEC);
   int VvsrsnbrDof = VvsrsFESpace->GetTrueVSize(); 
   cout << VvsrsnbrDof << " VFvsrsESpace degree of freedom\n";   
   {
      std::ofstream out("out/VvsrsFESpace.txt");
      VvsrsFESpace->Save(out);
   }

   //sub space for current over vsrs submesh.
   H1_FECollection *IvsrsFEC = new H1_FECollection(order, 1);
   FiniteElementSpace *IvsrsFESpace = new FiniteElementSpace(vsrsmesh, IvsrsFEC);
   int IvsrsnbrDof = IvsrsFESpace->GetTrueVSize(); 
   cout << IvsrsnbrDof << " IFvsrsESpace degree of freedom\n";   
   {
      std::ofstream out("out/IvsrsFESpace.txt");
      IvsrsFESpace->Save(out);
   }

   //sub space for current over t0mesh submesh.
   H1_FECollection *It0FEC = new H1_FECollection(order, 1);
   FiniteElementSpace *It0FESpace = new FiniteElementSpace(t0mesh, It0FEC);
   int It0nbrDof = It0FESpace->GetTrueVSize(); 
   cout << It0nbrDof << " It0FESpace degree of freedom\n";  
   {
      std::ofstream out("out/It0FESpace.txt");
      It0FESpace->Save(out);
   }
//
//Create the forms.
//
   //dv/dx
   //the forms for the equation with dV/dx.
   //BLF_dvdx implements the x dimension V space derivative,
   ConstantCoefficient one(100.0), Small(1e-30);
   MixedBilinearForm *MBLF_dvdx = new MixedBilinearForm(VFESpace /*trial*/, VFESpace /*test*/);
   MBLF_dvdx->AddDomainIntegrator(new DerivativeIntegrator(one, 0)); //x direction.
   //MBLF_dvdx->AddDomainIntegrator(new DerivativeIntegrator(one, 0)); //x direction.
   MBLF_dvdx->AddDomainIntegrator(new MixedScalarMassIntegrator(Small));
   MBLF_dvdx->Assemble();
   MBLF_dvdx->Finalize();
   cout << MBLF_dvdx->Height() << " MBLF_dvdx Height()." << endl;
   cout << MBLF_dvdx->Width() << " MBLF_dvdx Width()." << endl;
   assert(MBLF_dvdx->Height() == VnbrDof && MBLF_dvdx->Width() == VnbrDof);
   {
      std::ofstream out("out/MBLF_dvdx.txt");
      MBLF_dvdx->SpMat().PrintMatlab(out); // instead of Print()
   }

   //MBLF_IV implements the y dimension I derivative which is time and
   //the I.
   ConstantCoefficient CC_R(R);
   ConstantCoefficient CC_L(L);
   MixedBilinearForm *MBLF_IV = new MixedBilinearForm(IFESpace /*trial*/, VFESpace /*test*/);
   MBLF_IV->AddDomainIntegrator(new DerivativeIntegrator(CC_L, 1));
   MBLF_IV->AddDomainIntegrator(new MixedScalarMassIntegrator(CC_R));
   MBLF_IV->Assemble();
   MBLF_IV->Finalize();//Mixed Bilinear form VL1.
   cout << MBLF_IV->Height() << " MBLF_IV Height()." << endl;
   cout << MBLF_IV->Width() << " MBLF_IV Width()." << endl;
   assert(MBLF_IV->Height() == VnbrDof);
   assert(MBLF_IV->Width() == InbrDof);
   {
      std::ofstream out("out/MBLF_IV.txt");
      MBLF_IV->SpMat().PrintMatlab(out); // instead of Print()
   }


   //the forms for the equation with dI/dx.
   //MBLF_didx implements the x dimension I space derivative,
   Vector xDir(2); xDir = 0.0; xDir(0) = 1.0;
   VectorConstantCoefficient xDirCoeff(xDir);
   MixedBilinearForm *MBLF_didx = new MixedBilinearForm(IFESpace /*trial*/, IFESpace /*test*/);
   MBLF_didx->AddDomainIntegrator(new DerivativeIntegrator(one, 0)); //x direction.
   //MBLF_didx->AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(xDirCoeff)); //x direction.
   //MBLF_didx->AddDomainIntegrator(new DerivativeIntegrator(one, 0)); //x direction.
   MBLF_didx->AddDomainIntegrator(new MixedScalarMassIntegrator(Small));
   MBLF_didx->Assemble();
   MBLF_didx->Finalize();
   cout << MBLF_didx->Height() << " MBLF_didx Height()." << endl;
   cout << MBLF_didx->Width() << " MBLF_didx Width()." << endl;
   assert(MBLF_didx->Height() == InbrDof && MBLF_didx->Width() == InbrDof );
   {
      std::ofstream out("out/MBLF_didx.txt");
      MBLF_didx->SpMat().PrintMatlab(out); // instead of Print()
   }

   //MBLF_VI implements the y dimension V derivative which is time and
   //the V.
   ConstantCoefficient CC_G(G);
   ConstantCoefficient CC_C(C);
   
   //Vector vCy(2); vCy = 0.0; vCy(1) = C;
   //VectorConstantCoefficient CC_Cy(vCy);
   MixedBilinearForm *MBLF_VI = new MixedBilinearForm(VFESpace, IFESpace);
   MBLF_VI->AddDomainIntegrator(new DerivativeIntegrator(CC_C, 1));   //time derivative.
   MBLF_VI->AddDomainIntegrator(new MixedScalarMassIntegrator(CC_G));
   MBLF_VI->Assemble();
   MBLF_VI->Finalize();
   cout << MBLF_VI->Height() << " MBLF_VI Height()." << endl;
   cout << MBLF_VI->Width() << " MBLF_VI Width()." << endl;
   assert(MBLF_VI->Height() == InbrDof);
   assert(MBLF_VI->Width() == VnbrDof);
   {
      std::ofstream out("out/MBLF_VI.txt");
      MBLF_VI->SpMat().PrintMatlab(out); // instead of Print()
   }

   //Mixed Bilinear form VL1.
   MixedBilinearForm *MBLF_VL1 = new MixedBilinearForm(VvsrsFESpace /*trial*/, LM1FESpace /*test*/);
   MBLF_VL1->AddDomainIntegrator(new MixedScalarMassIntegrator(one));
   MBLF_VL1->Assemble();
   MBLF_VL1->Finalize();
   cout << MBLF_VL1->Height() << " MBLF_VL1 Height()." << endl;
   cout << MBLF_VL1->Width() << " MBLF_VL1 Width()." << endl;
   assert(MBLF_VL1->Height() == LM1nbrDof);
   assert(MBLF_VL1->Width() == VvsrsnbrDof);
   //create the submesh to parent operators for the Lagrange multiplier.
   SubmeshOperator *MBLF_VL1_map = new SubmeshOperator(MBLF_VL1->SpMat(), *VFESpace, *VvsrsFESpace, *LM1FESpace);
   TransposeOperator *MBLF_VL1_map_T = new TransposeOperator(MBLF_VL1_map);
   cout << MBLF_VL1_map->Height() << " MBLF_VL1_map Height()." << endl;
   cout << MBLF_VL1_map->Width() << " MBLF_VL1_map Width()." << endl;
   assert(MBLF_VL1_map->Height() == LM1nbrDof);
   assert(MBLF_VL1_map->Width() == VnbrDof);

   //Mixed Bilinear form IL1.
   MixedBilinearForm *MBLF_IL1 = new MixedBilinearForm(IvsrsFESpace /*trial*/, LM1FESpace /*test*/);
   FunctionCoefficient IL1Coeff(IL1Func);
   MBLF_IL1->AddDomainIntegrator(new MixedScalarMassIntegrator(IL1Coeff));
   MBLF_IL1->Assemble();
   MBLF_IL1->Finalize();
   cout << MBLF_IL1->Height() << " MBLF_IL1 Height()." << endl;
   cout << MBLF_IL1->Width() << " MBLF_IL1 Width()." << endl;
   assert(MBLF_IL1->Height() == LM1nbrDof);
   assert(MBLF_IL1->Width() == IvsrsnbrDof);
   //create the submesh tp parent operators for the Lagrange multiplier.
   SubmeshOperator *MBLF_IL1_map = new SubmeshOperator(MBLF_IL1->SpMat(), *IFESpace, *IvsrsFESpace, *LM1FESpace);
   TransposeOperator *MBLF_IL1_map_T = new TransposeOperator(MBLF_IL1_map);
   cout << MBLF_IL1_map->Height() << " MBLF_IL1_map Height()." << endl;
   cout << MBLF_IL1_map->Width() << " MBLF_IL1_map Width()." << endl;
   assert(MBLF_IL1_map->Height() == LM1nbrDof);
   assert(MBLF_IL1_map->Width() == InbrDof);

 
   {
      std::ofstream out("out/MBLF_VL1.txt");
      MBLF_VL1->SpMat().PrintMatlab(out); // instead of Print()
   }

   {
      std::ofstream out("out/MBLF_IL1.txt");
      MBLF_IL1->SpMat().PrintMatlab(out); // instead of Print()
   }

   //Mixed Bilinear form IL2.
   MixedBilinearForm *MBLF_IL2 = new MixedBilinearForm(It0FESpace /*trial*/, LM2FESpace /*test*/);
   ConstantCoefficient One(1.0);
   MBLF_IL2->AddDomainIntegrator(new MixedScalarMassIntegrator(One));
   MBLF_IL2->Assemble();
   MBLF_IL2->Finalize();
   cout << MBLF_IL2->Height() << " MBLF_IL2 Height()." << endl;
   cout << MBLF_IL2->Width() << " MBLF_IL2 Width()." << endl;
   assert(MBLF_IL2->Height() == LM2nbrDof);
   assert(MBLF_IL2->Width() == It0nbrDof);
   //create the submesh tp parent operators for the Lagrange multiplier.
   SubmeshOperator *MBLF_IL2_map = new SubmeshOperator(MBLF_IL2->SpMat(), *IFESpace, *It0FESpace, *LM2FESpace);
   TransposeOperator *MBLF_IL2_map_T = new TransposeOperator(MBLF_IL2_map);
   cout << MBLF_IL2_map->Height() << " MBLF_IL2_map Height()." << endl;
   cout << MBLF_IL2_map->Width() << " MBLF_IL2_map Width()." << endl;
   assert(MBLF_IL2_map->Height() == LM2nbrDof);
   assert(MBLF_IL2_map->Width() == InbrDof);
   
   {
      std::ofstream out("out/MBLF_IL2.txt");
      MBLF_IL2->SpMat().PrintMatlab(out); // instead of Print()
   }

   LinearForm *LFVS = new LinearForm(LM1FESpace);
   LFVS->AddDomainIntegrator(new DomainLFIntegrator(VsRsFunctionCoeff));
   LFVS->Assemble();

   cout << LFVS->Size() << " LFVS Size()." << endl;
 
   {
      std::ofstream out("out/lfvs.txt");
      LFVS->Print(out, 1); // instead of Print()
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
      Array<int> *BBrowOffset = new Array<int>(3); 
      (*BBrowOffset)[0]=0;
      (*BBrowOffset)[1]=MBLF_VL1_map->Height(); 
      (*BBrowOffset)[2]=MBLF_IL2_map->Height();
      //(*BBrowOffset)[3]=MBLF_IL3_map->Height();
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
      BB->SetBlock(1, 1, MBLF_IL2_map);
      //BB->SetBlock(2, 1, MBLF_IL3_map);
      
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

      // Create vector x.
      Vector x(OuterBlock->Height());
      x = 0.0;

      // Create vector b.
      Vector b(OuterBlock->Height());
      b = 0.0;
      b.SetVector(*LFVS, VnbrDof+InbrDof);

      BlockVector *bBlock = new BlockVector(*OuterBlockOffsets);
      *bBlock = 0.0;
      bBlock->GetBlock(1).SetVector(*LFVS, 0);
     
      {
         std::ofstream out("out/b.txt");
         b.Print(out, 1);
      }



//
// Prepare the preconditionner...
//
//   CompleteSchurPreconditioner *schur;
//   schur = new CompleteSchurPreconditioner(*OuterBlock);

//
// Solve the equations system.
//
   if(1)
   {
      FGMRESSolver solver;
      //solver.SetFlexible(true);
      solver.SetAbsTol(0);
      solver.SetRelTol(1e-2);
      solver.SetMaxIter(100);
      solver.SetPrintLevel(1);
      solver.SetKDim(20);
      solver.SetOperator(*OuterBlock);
      //solver.SetPreconditioner(*schur);
      solver.Mult(b, x);
      cout << solver.GetFinalRelNorm() << " GetFinalRelNorm" << endl;
   }

    // Assign xV and xI to refer to V and I.

      Vector xV(x, 0, VnbrDof);
      Vector xI(x, VnbrDof, InbrDof);

   GridFunction *VGF = new GridFunction(VFESpace, xV, 0);
   Glvis(mesh, VGF, "Voltage", 8, " keys ''");

   GridFunction *IGF = new GridFunction(IFESpace, xI, 0);
   Glvis(mesh, IGF, "Current", 8, " keys ''");


   return 0;
}



         