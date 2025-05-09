//                                stlt
//                                
// Compile with: make stlt, need MFEM version 4.7 and GLVIS-4.3.
//
// Sample runs:  ./stlt
//
/*
Description:  stlt (single transmission line transient) simulate a 
single transmission line with various soure signal,
source impedance and load impedance.

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
double R = 10e-3;  // Resistance per unit length
double G = 1.0e-9;  // Conductance per unit length

// source and load impedance.
double Rs = 50.0;   // source impedance.
double Rl = 50.0; //load impedance.

      real_t deltaT =0.01e-9;
      real_t endTime = 10e-9;
      real_t Time =0.0;

      Mesh *mesh;
      int dim; 
      int nbrel;  //number of element.

      FiniteElementCollection *VFEC, *IFEC;
      FiniteElementSpace *VFESpace, *IFESpace;
      int nbrVDof, nbrIDof;   //number of degree of freedom.

      MixedBilinearForm *M_V, *M_I;
      BilinearForm *S_V, *S_I;

      SparseMatrix *smM_V, *smM_I, *smS_V, *smS_I;
      SparseMatrix *K1smS_V, *K2smM_V, *K3smM_I, *K4smS_I;
      DenseMatrix *VDofByOne, *OneByOne;

      BlockOperator *lhsOp;
      Array<int> *lhsRowBlockOffset;
      Array<int> *lhsColBlockOffset;
      
      BlockOperator *rhsOp;
      Array<int> *rhsRowBlockOffset;
      Array<int> *rhsColBlockOffset;
      
      Vector *xL, *xR;
      
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
      int TimeSteps();
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
   return 1.0;
   if(t>1e-9) return 1.0;
   else return 0.0;
}

TransmissionLineTransient::TransmissionLineTransient()
{
   
}
 




int TransmissionLineTransient::TimeSteps()
{
   FunctionCoefficient SourceCoefficient(SourceFunction);
   

   xL = new Vector(nbrVDof+nbrIDof);
   *xL = 0.0;
   xR = new Vector(1+nbrVDof+nbrIDof);
   
   Vector zeroVector(3);
   zeroVector = 0.0;

   while(Time<endTime)
   {
      for(int i=1; i<1+nbrVDof+nbrIDof; i++) (*xR)[i]=(*xL)[i-1];
      (*xR)[0]=SourceFunction(zeroVector, Time);
      (*xL)=0.0;
      //rhs column y
      Vector y(nbrVDof+nbrIDof);
      rhsOp->Mult(*xR, y);
      {
         std::ofstream out("out/y.txt");
         if(printMatrix) y.Print(out, 10);
      }

      // solve lhsOp xL = xR, to get xR.
      GMRESSolver solver;
      solver.SetOperator(*lhsOp);
      //solver.SetPreconditioner(*block_prec);
      solver.SetRelTol(1e-12);
      //   solver.SetAbsTol(1e-8);
      solver.SetMaxIter(50000);
      solver.SetPrintLevel(1);
      solver.Mult(y, *xL);     
      {
         std::ofstream out("out/xL.txt");
         if(printMatrix) xL->Print(out, 10);
      }
   Time += deltaT;
   }

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
   VFEC = new H1_FECollection(order, dim);
   VFESpace = new FiniteElementSpace(mesh, VFEC);
   nbrVDof = VFESpace->GetNDofs(); 
   cout << "\nVFESpace\n" << VFESpace->GetNDofs() << " degree of freedom\n"
        << VFESpace->GetVDim() << " vectors dimension\n\n";   

   IFEC = new L2_FECollection(order, dim);
   IFESpace = new FiniteElementSpace(mesh, IFEC);
   nbrIDof = IFESpace->GetNDofs(); 
   cout << "\nIFESpace\n" << IFESpace->GetNDofs() << " degree of freedom\n"
        << IFESpace->GetVDim() << " vectors dimension\n\n";   

return 1;
}
/*
int TransmissionLineTransient::CreateEssentialBoundary()
{
   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   assert(mesh->bdr_attributes.Max()==5);
   ess_bdr.SetSize(mesh->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[output-1]=1;
   ess_bdr[stubend-1]=1;
   VFESpace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   {
      std::ofstream out("out/V_ess_tdof_list.txt");
      ess_tdof_list.Print(out, 10);
   }

   ess_bdr = 0;
   IFESpace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   {
      std::ofstream out("out/I_ess_tdof_list.txt");
      ess_tdof_list.Print(out, 10);
   }

   return 1;
}
*/
int TransmissionLineTransient::CreateBilinears()
{
   // Set up the mass and stiffness matrices
   

   M_V = new MixedBilinearForm(IFESpace, VFESpace);
   M_I = new MixedBilinearForm(VFESpace, IFESpace);
   S_V = new BilinearForm(VFESpace);
   S_I = new BilinearForm(IFESpace);

   // Define the mass and stiffnes
   ConstantCoefficient one(1.0);
 
   M_V->AddDomainIntegrator(new MixedScalarMassIntegrator(one));
   M_I->AddDomainIntegrator(new MixedScalarMassIntegrator(one));
   
   S_V->AddDomainIntegrator(new DerivativeIntegrator(one, 0));
   S_I->AddDomainIntegrator(new DerivativeIntegrator(one, 0));
    
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
   
   // modify matrix to consider the source impedance.
   K1smS_V->ScaleRow(0, 0.0);
   K2smM_V->ScaleRow(0, 0.0);
   K2smM_V->Set(0, 0, -Rs);

   // modify matrix to consider the load impedance.
   K1smS_V->ScaleRow(nbrVDof-1, 0.0);
   K2smM_V->ScaleRow(nbrVDof-1, 0.0);
   K2smM_V->Set(nbrVDof-1, nbrIDof-1, Rl);


   
   {
      std::ofstream out("out/M_V.txt");
      if(printMatrix) M_V->SpMat().Print(out, 10);
   }

   {
      std::ofstream out("out/M_I.txt");
      if(printMatrix) M_I->SpMat().Print(out, 10);
   }

   {
      std::ofstream out("out/S_V.txt");
      if(printMatrix) S_V->SpMat().Print(out, 10);
   }

   {
      std::ofstream out("out/S_I.txt");
      if(printMatrix) S_I->SpMat().Print(out, 10);
   }

   return 1;
}


int TransmissionLineTransient::CreateLhsBlockOperator()
{

// 6. Define the BlockStructure of lhsMatrix
   lhsRowBlockOffset = new Array<int>(3);
   (*lhsRowBlockOffset)[0]=0;
   (*lhsRowBlockOffset)[1]=nbrVDof; 
   (*lhsRowBlockOffset)[2]=nbrIDof; 
   lhsRowBlockOffset->PartialSum();
   {
      std::ofstream out("out/lhsrowblockOffset.txt");
      lhsRowBlockOffset->Print(out, 10);
   }
  
   lhsColBlockOffset = new Array<int>(3);
   (*lhsColBlockOffset)[0]=0;
   (*lhsColBlockOffset)[1]=nbrVDof; 
   (*lhsColBlockOffset)[2]=nbrIDof; 
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
   
   lhsOp->SetBlock(0, 1, smM_V);
   lhsOp->SetBlock(1, 0, smM_I);

      {
      std::ofstream out("out/lhsOp.txt");
      if(printMatrix) lhsOp->PrintMatlab(out);
      }

      assert(lhsOp->Height() == nbrIDof + nbrVDof);
      assert(lhsOp->Width() == nbrIDof + nbrVDof);

   return 1;
}


int TransmissionLineTransient::CreateRhsBlockOperator()
{

// 6. Define the BlockStructure of lhsMatrix
   rhsRowBlockOffset = new Array<int>(3);
   (*rhsRowBlockOffset)[0]=0;
   (*rhsRowBlockOffset)[1]=nbrVDof; 
   (*rhsRowBlockOffset)[2]=nbrIDof; 
   rhsRowBlockOffset->PartialSum();
   {
      std::ofstream out("out/rhsrowblockOffset.txt");
      rhsRowBlockOffset->Print(out, 10);
   }
  
   rhsColBlockOffset = new Array<int>(4);
   (*rhsColBlockOffset)[0]=0;
   (*rhsColBlockOffset)[1]=1;
   (*rhsColBlockOffset)[2]=nbrVDof; 
   (*rhsColBlockOffset)[3]=nbrIDof; 
   rhsColBlockOffset->PartialSum();
   {
      std::ofstream out("out/rhscolblockOffset.txt");
      rhsColBlockOffset->Print(out, 10);
   }
  
   Device device("cpu");
   MemoryType mt = device.GetMemoryType();

   VDofByOne = new DenseMatrix(nbrVDof, 1);
   VDofByOne->Elem(0, 0) = 1.0;

   OneByOne = new DenseMatrix(1, 1);
   OneByOne->Elem(0, 0) = 1.0;

   rhsOp = new BlockOperator(*rhsRowBlockOffset, *rhsColBlockOffset);
   
// Build the operator, insert each block.
   rhsOp->SetBlock(0, 0, VDofByOne);
   rhsOp->SetBlock(0, 1, K1smS_V);
   rhsOp->SetBlock(0, 2, K2smM_V);
   rhsOp->SetBlock(1, 1, K3smM_I);
   rhsOp->SetBlock(1, 2, K4smS_I);
   

      {
      std::ofstream out("out/rhsOp.txt");
      if(printMatrix) rhsOp->PrintMatlab(out);
      }

      assert(rhsOp->Height() == nbrIDof + nbrVDof);
      assert(rhsOp->Width() == 1+ nbrIDof + nbrVDof);
   
   return 1;
}


int TransmissionLineTransient::CleanOutDir()
{
    system("rm -f out/*");
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
