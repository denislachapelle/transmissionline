//                                stlt
//                                
// Compile with: make stltfe, need MFEM version 4.7 and GLVIS-4.3.
//
// Sample runs:  ./stltfe
//
/*
Description:  stltfe (single transmission line transient finite element) simulate a 
single transmission line with various source signals,
source impedances and load impedances.
*/

#include <mfem.hpp>
#include "mytools.hpp"
#include <iostream>
#include <math.h>
#include <filesystem>

//Physical group expected from mesh file.
#define tl1 1
#define input 2
#define output 3

using namespace std;
using namespace mfem;

class TransmissionLineTransient
{
   private:
      // 1. Parse command-line options.
      const char *meshFile = "stlmesh-1.msh";
      int order = 1;
      bool printMatrix = true;
      
// Constants for the telegrapher’s equation for RG-58, 50 ohm.
double L = 250e-9;  // Inductance per unit length
double C = 100.0e-12; // Capacitance per unit length
double R = 1000e-3;  // Resistance per unit length
double G = 1.0e-9;  // Conductance per unit length

// source and load impedance.
double Rs = 50.0;   // source impedance.
double Rl = 50.0; //load impedance.

      real_t deltaT = 0.001e-9;
      real_t endTime = 20e-9;
      real_t Time = 0.0;

      Mesh *mesh;
      int dim; 
      int nbrel;  //number of element.

      FiniteElementCollection *FEC;
      FiniteElementSpace *FESpace;
      int nbrDof;   //number of degree of freedom.

      BilinearForm *S_V, *S_I, *M_V, *M_I;

      SparseMatrix *smM_V, *smM_I, *smS_V, *smS_I;
      SparseMatrix *K1smS_V, *K2smM_V, *K3smM_I, *K4smS_I;
      DenseMatrix *DofByOne, *OneByOne;

      BlockOperator *lhsOp;
      Array<int> *lhsRowBlockOffset;
      Array<int> *lhsColBlockOffset;
      
      BlockOperator *rhsOp;
      Array<int> *rhsRowBlockOffset;
      Array<int> *rhsColBlockOffset;
      
      Vector *xL, *xR, *b, *Input;
      
      GSSmoother **gs;
      BlockDiagonalPreconditioner *block_prec;

      GridFunction *VGF, *IGF; 
      
   
   public:
      TransmissionLineTransient();
      //delete all files in out dir.
      int CleanOutDir();
      //parse the options.
      int Parser(int argc, char *argv[]);
      int LoadMeshFile();
      int CreateFESpace();
      int CreateBilinears();
      int CreateLhsBlockOperator();
      int CreateRhsBlockOperator();
      int CreatePreconditionner();
      int TimeSteps();
      int debugFileWrite();
      /*
      int CreateEssentialBoundary();
      
      
      int CreaterhsVector();
      int CreatexVector();
      int CreatePreconditionner();
      int Solver();
      int PostPrecessing();
      int DisplayResults();
      int Save();
   */
};

real_t SourceFunction(const Vector x, real_t t)
{
   real_t th = 10e-9;
   if(t<2*th) return t/th*(1-t/th) * sin(2*M_PI*1e9*t);
   else return 0.0;
}

TransmissionLineTransient::TransmissionLineTransient()
{
   
}
 

int TransmissionLineTransient::CleanOutDir()
{
    system("rm -f out/*");
    return 1;
}

 


int TransmissionLineTransient::Parser(int argc, char *argv[])
{

   OptionsParser args(argc, argv);
   args.AddOption(&meshFile, "-mf", "--meshfile",
                  "file to use as mesh file.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&printMatrix, "-prm", "--printmatrix", "-dnprm", "--donotprintmatrix",
                  "Print of not the matrix.");
                 
   args.Parse();

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, occ::, RAJA and OpenMP based on command line options.
   Device device("cpu");
   device.Print();

   if (args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   return 0;
}

int TransmissionLineTransient::LoadMeshFile()
{

// 3. Read the mesh from the given mesh file.
   mesh = new Mesh(meshFile, 1, 1);
   
   dim = mesh->Dimension();
   assert(dim==1); //this software works only for dimension 1.

   cout << mesh->bdr_attributes.Max() << " bdr attr max\n"
        << mesh->Dimension() << " dimensions\n"
        << mesh->SpaceDimension() << " space dimensions\n"
        << mesh->GetNV() << " vertices\n"
        << mesh->GetNE() << " elements\n"
        << mesh->GetNBE() << " boundary elements\n"
        << mesh->GetNEdges() << " edges\n";

   nbrel = mesh->GetNE();

   return 1;
}

int TransmissionLineTransient::CreateFESpace()
{
   FEC = new H1_FECollection(order, dim);
   FESpace = new FiniteElementSpace(mesh, FEC);
   nbrDof = FESpace->GetNDofs(); 
   cout << FESpace->GetNDofs() << " degree of freedom\n";   

return 1;
}

int TransmissionLineTransient::CreateBilinears()
{
   // Set up the mass and stiffness matrices
  
   M_V = new BilinearForm(FESpace);
   M_I = new BilinearForm(FESpace);
   S_V = new BilinearForm(FESpace);
   S_I = new BilinearForm(FESpace);

   // Define the mass and stiffnes
   ConstantCoefficient one(1.0);
 
   M_V->AddDomainIntegrator(new MassIntegrator(one));
   M_I->AddDomainIntegrator(new MassIntegrator(one));
   
   Vector xDirVector(1);
   xDirVector=0.0;
   xDirVector[0]=1.0;
   VectorConstantCoefficient xDirCoeff(xDirVector);

   S_V->AddDomainIntegrator(new ConvectionIntegrator(xDirCoeff));
   S_I->AddDomainIntegrator(new ConvectionIntegrator(xDirCoeff));
    
   M_V->Assemble();
   M_I->Assemble();
   S_V->Assemble();
   S_I->Assemble();

   M_V->Finalize();
   M_I->Finalize();
   S_V->Finalize();
   S_I->Finalize();

   cout << M_V->Height() << " M_V->Height()\n";
   cout << M_V->Width() << " M_V->Width()\n";
   cout << M_I->Height() << " M_I->Height()\n";
   cout << M_I->Width() << "M _I->Width()\n";
   cout << S_V->Height() << " S_V->Height()\n";
   cout << S_V->Width() << " S_V->Width()\n";
   cout << S_I->Height() << " S_I->Height()\n";
   cout << S_I->Width() << " S_I->Width()\n";
   
   smM_V = new SparseMatrix(M_V->SpMat());
   smM_I = new SparseMatrix(M_I->SpMat());
   smS_V = new SparseMatrix(S_V->SpMat());
   smS_I = new SparseMatrix(S_I->SpMat());

   K1smS_V = new SparseMatrix(*smS_V);
   *K1smS_V *= deltaT/L;

   K2smM_V = new SparseMatrix(*smM_V);
   *K2smM_V *= (1 - R * deltaT / L);

   K3smM_I = new SparseMatrix(*smM_I);
   *K3smM_I *= (1 - G * deltaT / C);

   K4smS_I = new SparseMatrix(*smS_I);
   *K4smS_I *= deltaT/C;
/*
   // modify matrix to consider the source impedance.
   K3smM_I->ScaleRow(0, 0.0);
   K4smS_I->ScaleRow(0, 0.0);
   K4smS_I->Set(0, 0, -Rs);

   // modify matrix to consider the load impedance.
   K3smM_I->ScaleRow(nbrDof-1, 0.0);
   K4smS_I->ScaleRow(nbrDof-1, 0.0);
   K4smS_I->Set(nbrDof-1, nbrDof-1, Rl);
*/
   {
      std::ofstream out("out/K1smS_V.txt");
      if(printMatrix) K1smS_V->PrintMatlab(out);
   }

   {
      std::ofstream out("out/K2smM_V.txt");
      if(printMatrix) K2smM_V->PrintMatlab(out);
   }

   {
      std::ofstream out("out/K3smM_I.txt");
      if(printMatrix) K3smM_I->PrintMatlab(out);
   }

   {
      std::ofstream out("out/K4smS_I.txt");
      if(printMatrix) K4smS_I->PrintMatlab(out);
   }


   
   {
      std::ofstream out("out/M_V.txt");
      if(printMatrix) M_V->SpMat().PrintMatlab(out);
   }

   {
      std::ofstream out("out/M_I.txt");
      if(printMatrix) M_I->SpMat().PrintMatlab(out);
   }

   {
      std::ofstream out("out/S_V.txt");
      if(printMatrix) S_V->SpMat().PrintMatlab(out);
   }

   {
      std::ofstream out("out/S_I.txt");
      if(printMatrix) S_I->SpMat().PrintMatlab(out);
   }

   return 1;
}


int TransmissionLineTransient::CreateLhsBlockOperator()
{

// 6. Define the BlockStructure of lhsMatrix
   lhsRowBlockOffset = new Array<int>(3);
   (*lhsRowBlockOffset)[0]=0;
   (*lhsRowBlockOffset)[1]=nbrDof; 
   (*lhsRowBlockOffset)[2]=nbrDof; 
   lhsRowBlockOffset->PartialSum();
   {
      std::ofstream out("out/lhsrowblockOffset.txt");
      lhsRowBlockOffset->Print(out, 10);
   }
  
   lhsColBlockOffset = new Array<int>(3);
   (*lhsColBlockOffset)[0]=0;
   (*lhsColBlockOffset)[1]=nbrDof; 
   (*lhsColBlockOffset)[2]=nbrDof; 
   lhsColBlockOffset->PartialSum();
   {
      std::ofstream out("out/lhscolblockOffset.txt");
      lhsColBlockOffset->Print(out, 10);
   }
  
   Device device("cpu");
   MemoryType mt = device.GetMemoryType();

   lhsOp = new BlockOperator(*lhsRowBlockOffset, *lhsColBlockOffset);
   
// Build the operator, insert each block.
// row 0 ...
   
   lhsOp->SetBlock(0, 0, smM_I);
   lhsOp->SetBlock(1, 1, smM_V);

      {
      std::ofstream out("out/lhsOp.txt");
      if(printMatrix) lhsOp->PrintMatlab(out);
      }

      assert(lhsOp->Height() == 2 * nbrDof);
      assert(lhsOp->Width() == 2 * nbrDof);

   return 1;
}


int TransmissionLineTransient::CreateRhsBlockOperator()
{

// 6. Define the BlockStructure of lhsMatrix
   rhsRowBlockOffset = new Array<int>(3);
   (*rhsRowBlockOffset)[0]=0;
   (*rhsRowBlockOffset)[1]=nbrDof; 
   (*rhsRowBlockOffset)[2]=nbrDof; 
   rhsRowBlockOffset->PartialSum();
   {
      std::ofstream out("out/rhsrowblockOffset.txt");
      rhsRowBlockOffset->Print(out, 10);
   }
  
   rhsColBlockOffset = new Array<int>(3);
   (*rhsColBlockOffset)[0]=0;
   //(*rhsColBlockOffset)[1]=1;
   (*rhsColBlockOffset)[1]=nbrDof; 
   (*rhsColBlockOffset)[2]=nbrDof; 
   rhsColBlockOffset->PartialSum();
   {
      std::ofstream out("out/rhscolblockOffset.txt");
      rhsColBlockOffset->Print(out, 10);
   }
  
   Device device("cpu");
   MemoryType mt = device.GetMemoryType();

   DofByOne = new DenseMatrix(nbrDof, 1);
   *DofByOne = 0.0;
   DofByOne->Elem(0, 0) = 1.0;

   OneByOne = new DenseMatrix(1, 1);
   OneByOne->Elem(0, 0) = 0.0;

   rhsOp = new BlockOperator(*rhsRowBlockOffset, *rhsColBlockOffset);
   
// Build the operator, insert each block.
  // rhsOp->SetBlock(0, 0, DofByOne);
   rhsOp->SetBlock(0, 0, K3smM_I);
   rhsOp->SetBlock(0, 1, K4smS_I);
   rhsOp->SetBlock(1, 0, K1smS_V);
   rhsOp->SetBlock(1, 1, K2smM_V);
   

      {
      std::ofstream out("out/rhsOp.txt");
      if(printMatrix) rhsOp->PrintMatlab(out);
      }
 
      assert(rhsOp->Height() == 2 * nbrDof);
      assert(rhsOp->Width() == 0 + 2 * nbrDof);
   
   return 1;
}


int TransmissionLineTransient::CreatePreconditionner()
{
/*

   // Create smoothers for diagonal blocks
   gs = new GSSmoother*[2];
   gs[0] = new GSSmoother(*smM_I); // Gauss-Seidel smoother
   gs[1] = new GSSmoother(*smM_V); // Gauss-Siedel smoother
   block_prec = new BlockDiagonalPreconditioner(*lhsRowBlockOffset);
   block_prec->SetDiagonalBlock(0, gs[0]);
   block_prec->SetDiagonalBlock(1, gs[1]);
*/
   return 1;
}




int TransmissionLineTransient::TimeSteps()
{
   cout << deltaT << " deltaT\n";
   cout << sqrt(L*C) << " sqrt(L*C)\n";
   
   FunctionCoefficient SourceCoefficient(SourceFunction);
   
   xL = new Vector(2 * nbrDof);
   *xL = 0.0;
   xR = new Vector(0 + 2 * nbrDof);
   *xR = 0.0;
   b = new Vector(2 * nbrDof);
   *b = 0.0;
   Input = new Vector(1000);
   *Input = 0.0; int j = 0;

   Vector zeroVector(3);
   zeroVector = 0.0;

   while(Time<endTime)
   {
      for(int i=0; i<0+2*nbrDof; i++) (*xR)[i]=(*xL)[i-0];
      (*xR)[0]=SourceFunction(zeroVector, Time);
      (*Input)[j++]=(*xR)[0];
      (*xL)=0.0;
      //rhs column y
      
      rhsOp->Mult(*xR, y);

      // solve lhsOp xL = xR, to get xR.
      GMRESSolver solver;
      //solver.SetKDim(100);
      solver.SetOperator(*lhsOp);
      //solver.SetPreconditioner(*block_prec);
      //solver.SetRelTol(1e-12);
      //   solver.SetAbsTol(1e-8);
      //solver.SetMaxIter(500);
      //solver.SetPrintLevel(1);
      solver.Mult(y, *xL);     
      
   Time += deltaT;
   }

   return 1;
}

int TransmissionLineTransient::debugFileWrite()
{
   {
      std::ofstream out("out/xL.txt");
      if(printMatrix) xL->Print(out, 1);
   }

   {
      std::ofstream out("out/y.txt");
      if(printMatrix) y->Print(out, 1);
   }

   {
      std::ofstream out("out/input.txt");
      if(printMatrix) Input->Print(out, 1);
   }

   return 1;


}

int main(int argc, char *argv[])
{

   TransmissionLineTransient TLT;

   TLT.Parser(argc, argv);
   TLT.CleanOutDir();
   TLT.LoadMeshFile();
   TLT.CreateFESpace();
   TLT.CreateBilinears();
   TLT.CreateLhsBlockOperator();
   TLT.CreateRhsBlockOperator();
   TLT.TimeSteps();
   /*
   TLT.CreateEssentialBoundary();
   
   TLT.TimeSteps();
   TLT.CreaterhsVector();
   TLT.CreatexVector();
   TLT.CreatePreconditionner();
   TLT.Solver();
   TLT.PostPrecessing();
   TLT.DisplayResults();
   TLT.Save();
   */
   return 0;
}
