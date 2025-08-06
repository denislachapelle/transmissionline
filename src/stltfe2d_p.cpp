//                                stltferk4
//                                
// Compile with: make stltfe2d, need MFEM version 4.7 and GLVIS-4.3.
//
// Sample runs:  ./stltfe2d
//
/*
Description:  stltfe (single transmission line transient finite element
2s) simulate a single transmission line with various source
signals, source impedances and load impedances. It treat time as a space dimension
to make use of MFEM capability.

Written by Denis Lachapelle.
*/

#include <mfem.hpp>
#include "mytools.hpp"
#include <iostream>
#include <math.h>
#include <filesystem>

using namespace std;
using namespace mfem;

// global variables
bool printMatrix = true; //control matrix saving to file.
bool noSource = true; // no code for the source enabled.

real_t SourceFunctionGaussianPulse(const Vector x)
{
   /* gaussian pulse of tw wide centered at tc.*/
   real_t t = x(1); //1 for y-axis.
   cout << "X\n";
   x.Print();
   real_t tw = 20e-9;
   real_t tc = 100e-9;
   if(t<2*tc) return 4.0 * t/(2.0*tc)*(1-t/(2.0*tc)) * exp(-pow(((t-tc)/tw), 2.0));
   else return 0.0;
}

real_t SourceFunctionStep(const Vector x, real_t t)
{
      //step.
      real_t tau = 30e-9;
      return 1.0 - exp(-t/tau);
}

real_t SourceFunctionSine(const Vector x, real_t t)
{
      return sin(2*M_PI*13e6*t);
}

FunctionCoefficient VsRsFunctionCoeff(SourceFunctionGaussianPulse);

class Stltfe2d
{
   private:
      int order = 1;
      double length = 10;
      int nbrSeg = 100;

         
   // Constants for the telegrapher’s equation for RG-58, 50 ohm.
   double L = 250e-9;  // Inductance per unit length
   double C = 100e-12; // Capacitance per unit length
   double R = 220e-3;  // Resistance per unit length
   double G = 1.0e-2;  // Conductance per unit length

   double Rs = 50.0;
   

      real_t deltaT = 1e-9;
      real_t endTime = 50e-9;
      real_t Time = 0.0;

      Mesh *mesh;
      SubMesh *ismesh;
      ParMesh *pmesh, *ispmesh;
      int dim; 
      int nbrEl;

      FiniteElementCollection *VFEC, *IFEC, *ISFEC;
      ParFiniteElementSpace *VFESpace, *IFESpace, *ISFESpace;
      int VnbrDof, InbrDof, ISnbrDof;  //number of degree of freedom.

      ParBilinearForm *BLF_dvdx, *BLF_didx;
      ParMixedBilinearForm *MBLF_IV, *MBLF_VI, *MBLF_VL, *MBLF_IL;
      ParLinearForm *LFVS;

      HypreParMatrix *HPM_BLF_dvdx, *HPM_BLF_didx;
      HypreParMatrix *HPM_MBLF_IV, *HPM_MBLF_VI, *HPM_MBLF_VL, *HPM_MBLF_IL;
      TransposeOperator *HPM_T_MBLF_VL, *HPM_T_MBLF_IL;


      SparseMatrix *MBLF_VLtsm, *MBLF_ILtsm;

      Array<int> *BlockOffset;
      BlockOperator *telegOp;

      BlockDiagonalPreconditioner *prec;
      Vector *diagVec;

      HypreParaSails *prec_V, *prec_I;
      DSmoother *prec_L;

      SparseMatrix *diagSparse;
      GSSmoother *smoother0, *smoother1;

      BlockVector *bBlock;
      BlockVector *xBlock;  //unknown block vector made of V and I.
      Vector *bV, *bI;
      Vector *xV, *xI, *xL;

      Vector *I, *V;
      Array<int> *xBlockOffsets; 

      HypreBoomerAMG *amg0, *amg1;
      
      GridFunction *VGF, *IGF; 
   
   public:
      Stltfe2d();
      //delete all files in out dir.
      int CleanOutDir();
      //parse the options.
      int Parser(int argc, char *argv[]);
      int CreateMeshFile();
      int CreateFESpace();
      int CreateForms();
      int CreateBlockOperator();
      int CreatePreconditioner();
      int SolveTelegraph();
      int Display();
  
};



Stltfe2d::Stltfe2d()
{
   
}
 

int Stltfe2d::CleanOutDir()
{
    system("rm -f out/*");
    return 1;
}

 


int Stltfe2d::Parser(int argc, char *argv[])
{

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&printMatrix, "-prm", "--printmatrix", "-dnprm", "--donotprintmatrix",
                  "Print of not the matrix.");
                 
   args.Parse();

   

   if (args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   return 0;
}

int Stltfe2d::CreateMeshFile()
{

   // Creates a serial 2D mesh, divided into n equal intervals of space in x and time in y.
   mesh = new Mesh();
   *mesh = Mesh::MakeCartesian2D(nbrSeg, static_cast<int>(endTime / deltaT),
                                  Element::QUADRILATERAL, false, length, endTime, false);
   
   //Convert the serial mesh to parralel mesh.
   pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);  // uses copy constructor
  // ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
  // delete mesh;
   pmesh->PrintInfo();
   pmesh->CheckElementOrientation(); // Optional but helpful

   //create a submesh, from serial mesh, of the left boundary.
   Array<int> is_bdr_attrs;
   is_bdr_attrs.Append(4); // Attribute ID=4 for the left boundary.
   ismesh = new SubMesh(SubMesh::CreateFromBoundary(*mesh, is_bdr_attrs));
   //set BdrAttribute to 0 meaning they are not boundary, necessary for LFVS.
   for (int i = 0; i < ismesh->GetNBE(); i++)
   {
        ismesh->SetBdrAttribute(i, 0);
   }
   ismesh->FinalizeMesh();
   cout << ismesh->Dimension() << " ismesh->Dimension()\n";
   ismesh->PrintInfo();

   {
      std::ofstream out("out/ismeshprint.txt");
      if(printMatrix) ismesh->Print(out);
   }

   ispmesh = new ParMesh(MPI_COMM_WORLD, *ismesh);
   cout << ispmesh->Dimension() << " ismesh->Dimension()\n";
   ispmesh->PrintInfo();



   return 1;
}

int Stltfe2d::CreateFESpace()
{
   //space for voltage.
   VFEC = new H1_FECollection(order, 2);
   VFESpace = new ParFiniteElementSpace(pmesh, VFEC, 1, Ordering::byNODES);
   VnbrDof = VFESpace->GetTrueVSize(); 
   cout << VnbrDof << " VFESpace degree of freedom\n";   

   //space for current.
   IFEC = new L2_FECollection(order, 2);
   IFESpace = new ParFiniteElementSpace(pmesh, IFEC, 1, Ordering::byNODES);
   InbrDof = IFESpace->GetTrueVSize(); 
   cout << InbrDof << " IFESpace degree of freedom\n";   
   
   //space for lagrange multiplier relate to Vs Rs,
   //which cause boundary cnditions on I(0,y).
   ISFEC = new L2_FECollection(order, 1);
   ISFESpace = new ParFiniteElementSpace(ispmesh, ISFEC, 1, Ordering::byNODES);
   ISnbrDof = ISFESpace->GetTrueVSize(); 
   cout << ISnbrDof << " ISFESpace degree of freedom\n";   

   return 1;
}

int Stltfe2d::CreateForms()
{
   //the forms for the voltage equations
   //BLF_dvdx implements the x dimension V space derivative,

   ConstantCoefficient one(1.0);
   BLF_dvdx = new ParBilinearForm(VFESpace);
   BLF_dvdx->AddDomainIntegrator(new DerivativeIntegrator(one, 0)); //x direction.
   BLF_dvdx->Assemble();
   BLF_dvdx->Finalize();
   cout << BLF_dvdx->Size() << " BLF_dvdx Size()." << endl;
   //MBLF_IV implements the y dimension I derivative which is time and
   //the I.
   ConstantCoefficient CC_L(L);
   ConstantCoefficient CC_R(R);
   MBLF_IV = new ParMixedBilinearForm(IFESpace, VFESpace);
   MBLF_IV->AddDomainIntegrator(new DerivativeIntegrator(CC_L, 1)); //y direction.
   MBLF_IV->AddDomainIntegrator(new MixedScalarMassIntegrator(CC_R));
   MBLF_IV->Assemble();
   MBLF_IV->Finalize();
   cout << MBLF_IV->Height() << " MBLF_IV Height()." << endl;
   cout << MBLF_IV->Width() << " MBLF_IV Width()." << endl;
   
    //the forms for the voltage equations
   //BLF_didx implements the x dimension I space derivative,
   BLF_didx = new ParBilinearForm(IFESpace);
   BLF_didx->AddDomainIntegrator(new DerivativeIntegrator(one, 0)); //x direction.
   BLF_didx->Assemble();
   BLF_didx->Finalize();
   cout << BLF_didx->Size() << " BLF_didx Size()." << endl;
   //MBLF_VI implements the y dimension V derivative which is time and
   //the V.
   ConstantCoefficient CC_C(C);
   ConstantCoefficient CC_G(G);
   MBLF_VI = new ParMixedBilinearForm(VFESpace, IFESpace);
   MBLF_VI->AddDomainIntegrator(new DerivativeIntegrator(CC_C, 1)); //y direction.
   MBLF_VI->AddDomainIntegrator(new MixedScalarMassIntegrator(CC_G));
   MBLF_VI->Assemble();
   MBLF_VI->Finalize();
   cout << MBLF_VI->Height() << " MBLF_VI Height()." << endl;
   cout << MBLF_VI->Width() << " MBLF_VI Width()." << endl;
   

   //Mixed Bilinear form VL.
   MBLF_VL = new ParMixedBilinearForm(VFESpace, ISFESpace);
   ConstantCoefficient oneOverRs(1.0/Rs);
   MBLF_VL->AddDomainIntegrator(new MassIntegrator(oneOverRs));
   MBLF_VL->Assemble();
   MBLF_VL->Finalize();
   cout << MBLF_VL->Height() << " MBLF_VL Height()." << endl;
   cout << MBLF_VL->Width() << " MBLF_VL Width()." << endl;
   

   //Mixed Bilinear form IL.
   MBLF_IL = new ParMixedBilinearForm(IFESpace, ISFESpace);
   MBLF_IL->AddDomainIntegrator(new MixedScalarMassIntegrator(one));
   MBLF_IL->Assemble();
   MBLF_IL->Finalize();
   cout << MBLF_IL->Height() << " MBLF_IL Height()." << endl;
   cout << MBLF_IL->Width() << " MBLF_IL Width()." << endl;
 
 
   
   {
      std::ofstream out("out/MBLF_VL.txt");
      if(printMatrix) MBLF_VL->SpMat().PrintMatlab(out); // instead of Print()
   }

   {
      std::ofstream out("out/MBLF_IL.txt");
      if(printMatrix) MBLF_IL->SpMat().PrintMatlab(out); // instead of Print()
   }

  
   
   LFVS = new ParLinearForm(ISFESpace);
   LFVS->AddDomainIntegrator(new DomainLFIntegrator(VsRsFunctionCoeff));
   LFVS->Assemble();

   cout << LFVS->Size() << " LFVS Size()." << endl;
 
   {
      std::ofstream out("out/lfvs.txt");
      if(printMatrix) LFVS->Print(out, 1); // instead of Print()
   }

   return 1;
}


int Stltfe2d::CreateBlockOperator()
{

   HPM_BLF_dvdx = BLF_dvdx->ParallelAssemble();
   HPM_BLF_didx = BLF_didx->ParallelAssemble();
   HPM_MBLF_IV = MBLF_IV->ParallelAssemble();
   HPM_MBLF_VI = MBLF_VI->ParallelAssemble();
   HPM_MBLF_IL = MBLF_IL->ParallelAssemble();
   HPM_MBLF_VL = MBLF_VL->ParallelAssemble();

   HPM_T_MBLF_VL = new TransposeOperator(HPM_MBLF_VL);
   HPM_T_MBLF_IL = new TransposeOperator(HPM_MBLF_IL);
   
   // 6. Define the BlockStructure.
      BlockOffset = new Array<int>(4);
      (*BlockOffset)[0]=0;
      (*BlockOffset)[1]=VFESpace->GetVSize(); 
      (*BlockOffset)[2]=IFESpace->GetVSize();
      (*BlockOffset)[3]=ISFESpace->GetVSize();
      BlockOffset->PartialSum();
      {
         std::ofstream out("out/BlockOffset.txt");
         BlockOffset->Print(out, 10);
      }

      telegOp = new BlockOperator(*BlockOffset);

      telegOp->SetBlock(0, 0, HPM_BLF_dvdx);
      telegOp->SetBlock(0, 1, HPM_MBLF_IV);
      telegOp->SetBlock(0, 2, HPM_T_MBLF_VL);

      telegOp->SetBlock(1, 0, HPM_MBLF_VI);
      telegOp->SetBlock(1, 1, HPM_BLF_didx);
      telegOp->SetBlock(1, 2, HPM_T_MBLF_IL);

      telegOp->SetBlock(2, 0, HPM_MBLF_VL);
      telegOp->SetBlock(2, 1, HPM_MBLF_IL);
       
      
   cout << telegOp->Height() << " telegOp->Height()" << endl;
   cout << telegOp->Width() << " telegOp->Width()" << endl;

      {
      std::ofstream out("out/telegop.txt");
      if(printMatrix) telegOp->PrintMatlab(out); // instead of Print()
      }
   
      bV =new Vector(VnbrDof); *bV = 0.0;
      bI =new Vector(InbrDof); *bI = 0.0;
      bBlock = new BlockVector(*BlockOffset);
      bBlock->GetBlock(0) = *bV;
      bBlock->GetBlock(1) = *bI;     
      bBlock->GetBlock(2) = *LFVS;

      xV =new Vector(VnbrDof); *xV = 0.0;
      xI =new Vector(InbrDof); *xI = 0.0;
      xL =new Vector(InbrDof); *xL = 0.0;
      xBlock = new BlockVector(*BlockOffset);
      xBlock->GetBlock(0) = *xV;
      xBlock->GetBlock(1) = *xI;     
      xBlock->GetBlock(2) = *xL;


      

   return 1;
}


int Stltfe2d::CreatePreconditioner()
{
   prec = new BlockDiagonalPreconditioner(*BlockOffset);
   prec_V = new HypreParaSails(*HPM_BLF_dvdx);
   prec_I = new HypreParaSails(*HPM_BLF_didx);

   prec_V->SetFilter(1e-3);         // instead of 0.1
   prec_V->SetSymmetry(0);     // keep it nonsymmetric

   prec_I->SetFilter(1e-3);         // instead of 0.1
   prec_I->SetSymmetry(0);     // keep it nonsymmetric

   diagVec = new Vector(ISFESpace->GetTrueVSize());
   diagVec->operator=(1.0);
   SparseMatrix *diagL = new SparseMatrix(*diagVec); // Diagonal identity
   diagL->Finalize();
   prec_L = new DSmoother(*diagL);

   prec->SetDiagonalBlock(0, prec_V);
   prec->SetDiagonalBlock(1, prec_I);
   prec->SetDiagonalBlock(2, prec_L);
   return 1;
}

/*
int Stltfe2d::CreatePreconditioner()
{
 // As per ex5p.cpp, 12. Construct the operators for preconditioner
   //
   //                 P = [ diag(M)         0         ]
   //                     [  0       B diag(M)^-1 B^T ]
   //
   //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
   //     pressure Schur Complement.
   HypreParMatrix *MinvBt = NULL;
   HypreParVector *Md = NULL;
   HypreParMatrix *S = NULL;
   Vector Md_PA;
   Solver *invM, *invS;

      Md = new HypreParVector(MPI_COMM_WORLD, M->GetGlobalNumRows(),
                              M->GetRowStarts());
      M->GetDiag(*Md);

      MinvBt = B->Transpose();
      MinvBt->InvScaleRows(*Md);
      S = ParMult(B, MinvBt);

      invM = new HypreDiagScale(*M);
      invS = new HypreBoomerAMG(*S);
   

   invM->iterative_mode = false;
   invS->iterative_mode = false;

   BlockDiagonalPreconditioner *darcyPr = new BlockDiagonalPreconditioner(
      block_trueOffsets);
   darcyPr->SetDiagonalBlock(0, invM);
   darcyPr->SetDiagonalBlock(1, invS);   return 1;
}
*/
int Stltfe2d::SolveTelegraph()
{
      GMRESSolver solver;
      //solver.SetAbsTol(1e-32);
      solver.SetRelTol(1e-8);
      solver.SetMaxIter(100);
      solver.SetPrintLevel(1);
      solver.SetKDim(1000);
      solver.SetOperator(*telegOp);
      solver.SetPreconditioner(*prec);
      solver.Mult(*bBlock, *xBlock);
   return 1;

}

int Stltfe2d::Display()
{
   VGF = new GridFunction(VFESpace, *xV, 0);
   Glvis(mesh, VGF, "Voltage");

return 1;
}




int main(int argc, char *argv[])
{

   // Initialize MPI and HYPRE.

   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
//   Device device(device_config);
//   if (myid == 0) { device.Print(); }


   Stltfe2d TL;

   TL.Parser(argc, argv);
   TL.CleanOutDir();
   TL.CreateMeshFile();
   TL.CreateFESpace();
   TL.CreateForms();
   TL.CreateBlockOperator();
   TL.CreatePreconditioner();
   TL.SolveTelegraph();
   TL.Display();
   
   return 0;
}
