//                                stltferk4
//                                
// Compile with: make stltferk4, need MFEM version 4.7 and GLVIS-4.3.
//
// Sample runs:  ./stltferk4
//
/*
Description:  stltfe (single transmission line transient finite element
runge kutta 4) simulate a single transmission line with various source
signals, source impedances and load impedances.
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


real_t SourceFunctionGaussianPulse(const Vector x, real_t t)
{
   /* gaussian pulse of tw wide centered at tc.*/
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




/*
generate a time dependent signal
*/
real_t SourceFunction(const Vector x, real_t t)
{
   return SourceFunctionStep(x, t);
}


// TelegrapherOperator is used to timestep.
class TelegrapherOperator : public TimeDependentOperator
{
private:
   const Operator *lhsBlock, *rhsBlock;
   const int size;
   
public:
   TelegrapherOperator(const Operator *lhsBlock_, BlockDiagonalPreconditioner block_prec_, const Operator *rhsBlock_, int size_)
      : TimeDependentOperator(size_), size(size_), lhsBlock(lhsBlock_), block_prec(block_prec_), rhsBlock(rhsBlock_)
   {}

   virtual void Mult(const Vector &x, Vector &y) const
   {
      // x is const need to copy it to then force s[0].
      Vector s(x.Size());
      s=x;
      Vector Zero(3); 
      Zero = 0.0;
      s[size/2] += SourceFunction(Zero, t)/Rs;
      Vector b(x.Size());
      rhsBlock->Mult(s, b);

      GMRESSolver gmres;
      gmres.SetAbsTol(1e-12);
      gmres.SetRelTol(1e-8);
      gmres.SetMaxIter(200);
      gmres.SetPrintLevel(1);
      gmres.SetKDim(30); // Optional: restart after 30 iterations
      gmres.SetOperator(*lhsBlock);
      gmres.SetPreconditioner(block_prec);
      gmres.Mult(b, y);


   }
};
 

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
double R = 220e-3;  // Resistance per unit length
double G = 1.0e-9;  // Conductance per unit length
double lenght = 100;
int nbrSeg = 100;

// source and load impedance.
double Rs = 50.0;   // source impedance.
double Rl = 50.0; //load impedance.

      real_t deltaT = 0.01e-9;
      real_t endTime = 300e-9;
      real_t Time = 0.0;

      Mesh *mesh;
      int dim; 
      int nbrel;  //number of element.

      FiniteElementCollection *FEC;
      FiniteElementSpace *FESpace;
      int nbrDof;   //number of degree of freedom.

      BilinearForm *S_I, *S_V, *M;

      SparseMatrix *smM, *smS_I, *smS_V;

      BlockOperator *lhsOp;
      Array<int> *lhsBlockOffset;
            
      BlockOperator *rhsOp;
      Array<int> *rhsBlockOffset;

      TelegrapherOperator *teleOp;

      Vector *x, *xR, *b, *Input, *y;
      
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
      
};



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

/// Creates 1D mesh, divided into n equal intervals.
   mesh = new Mesh();
   *mesh = mfem::Mesh::MakeCartesian1D(nbrSeg, lenght);
   
   //mesh = new Mesh(meshFile, 1, 1);
   
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
   cout << nbrDof << " degree of freedom\n";   

return 1;
}

int TransmissionLineTransient::CreateBilinears()
{
   // Set up the mass and stiffness matrices

   // Define the mass and stiffnes
   ConstantCoefficient one(1.0);

   M = new BilinearForm(FESpace);
   M->AddDomainIntegrator(new MassIntegrator(one));
   M->Assemble();
   M->Finalize();

   S_I = new BilinearForm(FESpace);
   S_I->AddDomainIntegrator(new DerivativeIntegrator(one, 0));
   S_I->Assemble();
   S_I->Finalize();

   S_V = new BilinearForm(FESpace);
   S_V->AddDomainIntegrator(new DerivativeIntegrator(one, 0));
   S_V->Assemble();
   S_V->Finalize();

   cout << M->Height() << " M->Height()\n";
   cout << M->Width() << " M->Width()\n";
   
   cout << S_V->Height() << " S_V->Height()\n";
   cout << S_V->Width() << " S_V->Width()\n";

   cout << S_I->Height() << " S_I->Height()\n";
   cout << S_I->Width() << " S_I->Width()\n";
   
   smM = new SparseMatrix(M->SpMat());
   smS_V = new SparseMatrix(S_V->SpMat());
   //smS_V->Set(0, 0, 1.0/Rs);
   //smS_V->Set(0, nbrDof-1, 1.0/Rl);

   if(EntryExists(*smS_V, 0, 0))
   {
      smS_V->Add(0, 0, 1.0/Rs);
   }
   else
   {
      smS_V->Set(0, 0, 1.0/Rs);   
   }

   if(EntryExists(*smS_V, nbrDof-1, nbrDof-1))
   {
      smS_V->Add(nbrDof-1, nbrDof-1, 1.0/Rl);
   }
   else
   {
      smS_V->Set(nbrDof-1, nbrDof-1, 1.0/Rl);   
   }


   smS_I = new SparseMatrix(S_I->SpMat());
 
   smM->Finalize();
   smS_V->Finalize();
   smS_I->Finalize();
   
   {
      std::ofstream out("out/smS_V.txt");
      if(printMatrix) smS_V->PrintMatlab(out);
   }

   {
      std::ofstream out("out/smS_I.txt");
      if(printMatrix) smS_I->PrintMatlab(out);
   }

   {
      std::ofstream out("out/smM.txt");
      if(printMatrix) smM->PrintMatlab(out);
   }

   return 1;
}


int TransmissionLineTransient::CreateLhsBlockOperator()
{

// 6. Define the BlockStructure of lhsMatrix
   lhsBlockOffset = new Array<int>(3);
   (*lhsBlockOffset)[0]=0;
   (*lhsBlockOffset)[1]=nbrDof; 
   (*lhsBlockOffset)[2]=nbrDof; 
   lhsBlockOffset->PartialSum();
   {
      std::ofstream out("out/lhsBlockOffset.txt");
      lhsBlockOffset->Print(out, 10);
   }
  
   Device device("cpu");
   MemoryType mt = device.GetMemoryType();

   lhsOp = new BlockOperator(*lhsBlockOffset);
   
// Build the operator, insert each block.
// row 0 ...
   
   lhsOp->SetBlock(0, 0, smM, C);
   lhsOp->SetBlock(1, 1, smM, L);

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
   rhsBlockOffset = new Array<int>(3);
   (*rhsBlockOffset)[0]=0;
   (*rhsBlockOffset)[1]=nbrDof; 
   (*rhsBlockOffset)[2]=nbrDof; 
   rhsBlockOffset->PartialSum();
   {
      std::ofstream out("out/rhsBlockOffset.txt");
      rhsBlockOffset->Print(out, 10);
   }

   Device device("cpu");
   MemoryType mt = device.GetMemoryType();

   rhsOp = new BlockOperator(*rhsBlockOffset);
   
// Build the operator, insert each block.
  // rhsOp->SetBlock(0, 0, DofByOne);
   rhsOp->SetBlock(0, 0, smM, G);
   rhsOp->SetBlock(0, 1, smS_I, -1.0);
   rhsOp->SetBlock(1, 0, smS_V, -1.0);
   rhsOp->SetBlock(1, 1, smM, R);
   

      {
      std::ofstream out("out/rhsOp.txt");
      if(printMatrix) rhsOp->PrintMatlab(out);
      }
 
      assert(rhsOp->Height() == 2 * nbrDof);
      assert(rhsOp->Width() == 2 * nbrDof);
   
   return 1;
}


int TransmissionLineTransient::CreatePreconditionner()
{

      // 4. Optional: create a block preconditioner
      Array<int> blockOffsets(0, size/2);
      BlockDiagonalPreconditioner block_prec(blockOffsets);
      block_prec.SetDiagonalBlock(0, new GSSmoother(lhsBlock->GetBlock(0, 0)));
      block_prec.SetDiagonalBlock(1, new GSSmoother(lhsBlock->GetBlock(1, 1)));

   return 1;
}




int TransmissionLineTransient::TimeSteps()
{
   cout << deltaT << " deltaT\n";
   cout << sqrt(L*C) << " sqrt(L*C)\n";
   
   //FunctionCoefficient SourceCoefficient(SourceFunction);
   
   teleOp = new TelegrapherOperator(lhsOp, block_prec, rhsOp, 2 * nbrDof);
   
   RK4Solver solver;
   solver.Init(*teleOp);

   x = new Vector(2*nbrDof);
   *x = 0.0;

   int nbrPlot=5;
   real_t plotTime = endTime/nbrPlot;
   int plotCount = 0;


   while(Time<endTime)
   {
     
      solver.Step(*x, Time, deltaT);
         
           if(Time >= plotCount * plotTime)
      {
         std::string s = "out/x" + std::to_string(plotCount) + ".txt";
         std::ofstream out(s);
         Vector val(x->GetData(), nbrDof);
         val.Print(out, 1);
         plotCount++;
         cout << Time << " Time" << endl;
         cout << val.Max() << " Max" << endl;
      }

      Time += deltaT;
   } 
   
   
   {
      std::ostringstream oss;
       oss << "octave --persist --quiet --eval \"figure(1); hold on; ";
    
      for(int i=1; i<nbrPlot; i++) 
      {
         oss << "subplot(5,1, " << i << "); " << "plot(x" << i << "=load('out/x" << i << ".txt')); ";
      }
      oss << "input('Press Enter to close'); \"";
      std::string result = oss.str();
      system(result.c_str());
      
   //system("gnuplot -persist -e \"plot 'out/x1.txt' with lines title 'x1', \
                                      'out/x2.txt' with lines title 'x2', \
                                      'out/x3.txt' with lines title 'x3', \
                                      'out/x4.txt' with lines title 'x4'\"");
   }


   return 1;
}

int TransmissionLineTransient::debugFileWrite()
{
   {
      std::ofstream out("out/x.txt");
      if(printMatrix) x->Print(out, 1);
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

   return 0;
}
