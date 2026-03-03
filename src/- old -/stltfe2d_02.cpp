//                                stltfe2d_02.cpp
//                                
// Compile with: make stltfe2d_02, need MFEM version 4.8 and GLVIS-4.3.
//
// Sample runs:  ./stltfe2d_02
//
/*
Description:  stltfe_submesh(single transmission line transient finite element
2D) simulate a single transmission line with various source
signals, source impedances and load impedances. It treat time as a space dimension (y)
to make use of MFEM capability.
 
Written by Denis Lachapelle, December 2025
*/

#include <mfem.hpp>
#include "mytools.hpp"
#include <iostream>
#include <math.h>
#include <filesystem>

using namespace std;
using namespace mfem;

real_t lineLength = 200;                     // transmission line lenght.
real_t endTime = 1000e-9;              // Unscale End time.
real_t timeScaling;                    // make time and space axis the same size (space size).
    
int nbrLengthSeg = 400;
int nbrTimeSeg = 400;

real_t Rs = 50.0;
real_t Rl = 50.0;

real_t penalty = 1.0;


class LowerTriPrec : public mfem::Solver
{
private:
   const mfem::Array<int> &off;
   const mfem::Operator &A21;
   mfem::Solver &Inv11;
   mfem::Solver &Inv22;

   mutable mfem::Vector xV, xI, yV, yI, tmpI;

public:
   LowerTriPrec(const mfem::Array<int> &offsets,
                const mfem::Operator &A21_,
                mfem::Solver &Inv11_,
                mfem::Solver &Inv22_)
      : mfem::Solver(offsets.Last()),
        off(offsets), A21(A21_), Inv11(Inv11_), Inv22(Inv22_),
        xV(off[1]-off[0]), xI(off[2]-off[1]),
        yV(off[1]-off[0]), yI(off[2]-off[1]),
        tmpI(off[2]-off[1])
   {
      // Preconditioners are typically not iterative; leave default iter controls.
      iterative_mode = false;
   }

   void SetOperator(const mfem::Operator &op) override
   {
      // Not used: we already have references to the blocks we need.
      (void)op;
   }

   void Mult(const mfem::Vector &x, mfem::Vector &y) const override
   {
      // Split x = [xV; xI]
      xV.MakeRef(const_cast<mfem::Vector&>(x), off[0], off[1]-off[0]);
      xI.MakeRef(const_cast<mfem::Vector&>(x), off[1], off[2]-off[1]);

      // yV = A11^{-1} xV
      Inv11.Mult(xV, yV);

      // tmpI = xI - A21*yV
      A21.Mult(yV, tmpI);
      tmpI.Neg(); tmpI += xI;

      // yI = A22^{-1} tmpI
      Inv22.Mult(tmpI, yI);

      // Pack y = [yV; yI]
      y.SetSize(off.Last());
      mfem::Vector y0, y1;
      y0.MakeRef(y, off[0], off[1]-off[0]); y0 = yV;
      y1.MakeRef(y, off[1], off[2]-off[1]); y1 = yI;
   }
};

static real_t SourceFunctionGaussianPulse(const Vector x)
{
   /* gaussian pulse of tw wide centered at tc.*/
   real_t t = x(1); //1 for y-axis.
   real_t tw = 50e-9 * timeScaling;
   real_t tc = 100e-9 * timeScaling;
   if(t<2*tc) return 4.0 * t/(2.0*tc)*(1-t/(2.0*tc)) * exp(-pow(((t-tc)/tw), 2.0));
   else return 0.0;
}

real_t SourceFunctionStep(const Vector x)
{
      //step.
      real_t t = x(1); //1 for y-axis.
      real_t tau = 30e-9 * timeScaling;
      real_t td = 100e-9 *timeScaling;
      real_t out;
      if(t-td >= 0) out = 1.0 - exp(-(t-td)/tau);
      else out = 0.0;
      return out;
}

real_t SourceFunctionSine(const Vector x)
{
   real_t t = x[1];
   real_t freq = 10e6 / timeScaling;
   return sin(2*M_PI*freq*t);
}

real_t SourceFunction(const Vector x)
{
   real_t tt = x[1];
   real_t xx = x[0];
   real_t out;
   if(tt == 0.0) out = 0.0;
   else out = SourceFunctionGaussianPulse(x);
   return out*penalty;
}

FunctionCoefficient VsFunctionCoeff(SourceFunction);

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
   args.AddOption(&Rs, "-rs", "--rs",
               "Source resistance Rs (ohms).");

   args.AddOption(&Rl, "-rl", "--rl",
               "Load resistance Rl (ohms).");

   args.AddOption(&lineLength, "-len", "--length",
               "Transmission line length (meters).");

   args.AddOption(&endTime, "-T", "--t-final",
               "End time (seconds) (y-direction extent in the space-time mesh).");
   args.Parse();

   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   timeScaling = lineLength/endTime;

   
   // Constants for the telegrapher’s equation for RG-58, 50 ohm.
real_t L = 250e-9; //= 250e-9;  // Inductance per unit length
real_t C = 100e-12; //= 100e-12; // Capacitance per unit length
real_t R = 0.220;  // Resistance per unit length
real_t G = 1.0e-7;  // Conductance per unit length

   
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
                                  Element::QUADRILATERAL, true, lineLength, endTime * timeScaling, false);
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
   // Create the spaces.
   //

   //space for voltage.
   H1_FECollection *VFEC = new H1_FECollection(order, 2);
   FiniteElementSpace *VFESpace = new FiniteElementSpace(mesh, VFEC);
   int VnbrDof = VFESpace->GetVSize(); 
   cout << VnbrDof << " VFESpace degree of freedom\n";  

   {
      std::ofstream out("out/vfespace.txt");
      VFESpace->Save(out);
   }
   
   //space for current.
   H1_FECollection *IFEC = new H1_FECollection(order, 2);
   FiniteElementSpace *IFESpace = new FiniteElementSpace(mesh, IFEC);
   int InbrDof = IFESpace->GetVSize(); 
   cout << InbrDof << " IFESpace degree of freedom\n";   

   {
      std::ofstream out("out/ifespace.txt");
      IFESpace->Save(out);
   }
   
   Array<int> leftBdrMarker({0, 0, 0, 1});
   
   Array<int> rightBdrMarker({0, 1, 0, 0});
   
   Array<int> bottomBdrMarker({1, 0, 0, 0});

   Array<int> topBdrMarker({0, 0, 1, 0});


ConstantCoefficient leftBoundaryVConstCoeff(1.0 * penalty);
ConstantCoefficient rightBoundaryVConstCoeff(-1.0 * penalty);


//
//Create the forms.
//
   // A11
   ConstantCoefficient Small(1e-2), CC_ONE(1.0);
   Vector VxDir{1.0, 0.0};                 // C++11 initializer_list constructor
   VectorConstantCoefficient VCCxDir(VxDir);
   MixedBilinearForm *A11 = new MixedBilinearForm(VFESpace, IFESpace);
   A11->AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(VCCxDir)); //x direction.
   A11->AddBoundaryIntegrator(new BoundaryMassIntegrator(leftBoundaryVConstCoeff), leftBdrMarker);
   //A11->AddBoundaryIntegrator(new BoundaryMassIntegrator(rightBoundaryVConstCoeff), rightBdrMarker);
   //A11->AddDomainIntegrator(new MassIntegrator(Small));
   A11->Assemble();
   A11->Finalize();
   cout << A11->Height() << " A11 Height()." << endl;
   cout << A11->Width() << " A11 Width()." << endl;
   assert(A11->Height() == VnbrDof);
   assert(A11->Width() == VnbrDof);
     

   // A12 
   Vector vL{0.0, L*timeScaling};                 // C++11 initializer_list constructor
   VectorConstantCoefficient VCC_L(vL);
   ConstantCoefficient CC_R(R);
   ConstantCoefficient CC_Rs(Rs*penalty);
   ConstantCoefficient CC_Rl(Rl*penalty);
   MixedBilinearForm *A12 = new MixedBilinearForm(IFESpace /*trial*/, VFESpace /*test*/);
   A12->AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(VCC_L));
   A12->AddDomainIntegrator(new MixedScalarMassIntegrator(CC_R));
   A12->AddBoundaryIntegrator(new BoundaryMassIntegrator(CC_Rs), leftBdrMarker);
   //A12->AddBoundaryIntegrator(new BoundaryMassIntegrator(CC_Rl), rightBdrMarker);
   A12->Assemble();
   A12->Finalize();//Mixed Bilinear form VL1.
   cout << A12->Height() << " A12 Height()." << endl;
   cout << A12->Width() << " A12 Width()." << endl;
   assert(A12->Height() == VnbrDof);
   assert(A12->Width() == InbrDof);



   //A21
   Vector vC{0.0, C * timeScaling};                 // C++11 initializer_list constructor
   VectorConstantCoefficient VCC_C(vC);
   ConstantCoefficient CC_G(G);
   MixedBilinearForm *A21 = new MixedBilinearForm(VFESpace /*trial*/, IFESpace /*test*/);
   A21->AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(VCC_C)); //y direction.
   A21->AddDomainIntegrator(new MixedScalarMassIntegrator(CC_G));
   A21->Assemble();
   A21->Finalize();
   cout << A21->Height() << " A21 Height()." << endl;
   cout << A21->Width() << " A21 Width()." << endl;
   assert(A21->Height() == InbrDof);
   assert(A21->Width() == VnbrDof );

   //A22
   BilinearForm *A22 = new BilinearForm(IFESpace);
   A22->AddDomainIntegrator(new DerivativeIntegrator(CC_ONE, 0));   //x direction.
   A22->AddDomainIntegrator(new MassIntegrator(Small));
   A22->Assemble();
   A22->Finalize();
   cout << A22->Height() << " A22 Height()." << endl;
   cout << A22->Width() << " A22 Width()." << endl;
   assert(A22->Height() == InbrDof);
   assert(A22->Width() == InbrDof);
 

   
   //
   // the linearform
   //
   LinearForm *LFVS = new LinearForm(VFESpace);
   LFVS->AddBoundaryIntegrator(new BoundaryLFIntegrator(VsFunctionCoeff), leftBdrMarker);
   LFVS->Assemble();

   cout << LFVS->Size() << " LFVS Size()." << endl;
 
   {
      std::ofstream out("out/LFVS.txt");
      LFVS->Print(out, 1); // instead of Print()
   }

   
   //
   //create the x and b vectors.
   //   

      // Create vector x.
      Vector x(A11->Height() + A22->Height());
      x = 0.0;

      
      Vector xV(x, 0, A11->Height());
      Vector xI(x, A11->Height(), A22->Height());
    
      // Create vector b.
      Vector b(A11->Height() + A22->Height());
      b = 0.0;
      b.SetVector(*LFVS, 0);

      Vector bV(b, 0, A11->Height());
      Vector bI(b, A11->Height(), A22->Height());
       
      {
         std::ofstream out("out/b.txt");
         b.Print(out, 1);
      }

         {
mfem::Vector d;
A11->SpMat().GetDiag(d);
double minabs = std::numeric_limits<double>::infinity();
for (int i = 0; i < d.Size(); i++) { minabs = std::min(minabs, std::abs(d[i])); }
std::cout << "A11 min |diag| = " << minabs << std::endl;

A22->SpMat().GetDiag(d);
minabs = std::numeric_limits<double>::infinity();
for (int i = 0; i < d.Size(); i++) { minabs = std::min(minabs, std::abs(d[i])); }
std::cout << "A22 min |diag| = " << minabs << std::endl;
   }
/*
//
// arrange for the zero boundary value.
//
   Array<int> VessDofsMarker, IessDofsMarker;
   Array<int> VtopDofsMarker, ItopDofsMarker;   
   Array<int> VessDofsList, IessDofsList;
   Array<int> VtopDofsList, ItopDofsList;

// Get essential VDOFs (NOT true dofs) for mixed elimination
VFESpace->GetEssentialVDofs(bottomBdrMarker, VessDofsMarker);
IFESpace->GetEssentialVDofs(bottomBdrMarker, IessDofsMarker);
VFESpace->GetEssentialVDofs(topBdrMarker, VtopDofsMarker);
IFESpace->GetEssentialVDofs(topBdrMarker, ItopDofsMarker);

VFESpace->MarkerToList(VessDofsMarker, VessDofsList);
IFESpace->MarkerToList(IessDofsMarker, IessDofsList);
VFESpace->MarkerToList(VtopDofsMarker, VtopDofsList);
IFESpace->MarkerToList(ItopDofsMarker, ItopDofsList);


   {
      std::ofstream out("out/VessDofsList.txt");
      VessDofsList.Print(out);
   }

   {
      std::ofstream out("out/IessDofsList.txt");
      IessDofsList.Print(out);
   }
   

// Diagonal blocks: set diag = 1 and zero row/col
A11->EliminateEssentialBCFromDofsDiag(VessDofsMarker, 1.0);
A22->EliminateEssentialBCFromDofsDiag(IessDofsMarker, 1.0);
A11->EliminateEssentialBCFromDofsDiag(VtopDofsMarker, 1.0);
A22->EliminateEssentialBCFromDofsDiag(ItopDofsMarker, 1.0);


// Eliminate trial columns (and update RHS if needed)
// If prescribed values are zero, you can just eliminate columns without RHS update.
// The MFEM call that does the bookkeeping is EliminateTrialVDofs(vdofs, sol, rhs).
// Here solV/solI are the prescribed vectors in those spaces (often zero).
A21->EliminateTrialVDofs(VessDofsList, xV, bI); // removes V columns from (I,V), updates bI
A12->EliminateTrialVDofs(IessDofsList, xI, bV); // removes I columns from (V,I), updates bV
A21->EliminateTrialVDofs(VtopDofsList, xV, bI); // removes V columns from (I,V), updates bI
A12->EliminateTrialVDofs(ItopDofsList, xI, bV); // removes I columns from (V,I), updates bV

A12->EliminateTestVDofs(VessDofsList);
A21->EliminateTestVDofs(IessDofsList);
A12->EliminateTestVDofs(VtopDofsList);
A21->EliminateTestVDofs(ItopDofsList);
*/

   {
      std::ofstream out("out/A11.txt");
      A11->SpMat().PrintMatlab(out); // instead of Print()
   }

   {
      std::ofstream out("out/A12.txt");
      A12->SpMat().PrintMatlab(out); // instead of Print()
   }
   
   {
      std::ofstream out("out/A21.txt");
      A21->SpMat().PrintMatlab(out); // instead of Print()
   }

   {
      std::ofstream out("out/A22.txt");
      A22->SpMat().PrintMatlab(out); // instead of Print()
   }

   {
mfem::Vector d;
A11->SpMat().GetDiag(d);
double minabs = std::numeric_limits<double>::infinity();
for (int i = 0; i < d.Size(); i++) { minabs = std::min(minabs, std::abs(d[i])); }
std::cout << "A11 min |diag| = " << minabs << std::endl;

A22->SpMat().GetDiag(d);
minabs = std::numeric_limits<double>::infinity();
for (int i = 0; i < d.Size(); i++) { minabs = std::min(minabs, std::abs(d[i])); }
std::cout << "A22 min |diag| = " << minabs << std::endl;
   }
//
//Define the inner and outer blockopertor.
//


      //
      // create the block structure.
      //
           
      
      Array<int> *blockOffsets = new Array<int>(3);
      (*blockOffsets)[0]=0;
      (*blockOffsets)[1]=A11->Height(); 
      (*blockOffsets)[2]=A22->Height();
      blockOffsets->PartialSum();
      
      {
         std::ofstream out("out/blockOffsets.txt");
         blockOffsets->Print(out, 10);
      }

      BlockOperator *block = new BlockOperator(*blockOffsets);

      block->SetBlock(0, 0, A11);
      block->SetBlock(0, 1, A12);
      block->SetBlock(1, 0, A21);
      block->SetBlock(1, 1, A22);

      cout << block->Height() << " block->Height()" << endl;
      cout << block->Width() << " block->Width()" << endl;

      {
      std::ofstream out("out/block.txt");
      if(printMatrix) block->PrintMatlab(out); // instead of Print()
      }



//
// Prepare the preconditionner...
//

#ifdef MFEM_USE_SUITESPARSE
  // OK: class exists
#else
  #error "MFEM was built without SuiteSparse. Rebuild MFEM with MFEM_USE_SUITESPARSE=YES."
#endif

auto *S11 = new mfem::UMFPackSolver();
S11->SetOperator(A11->SpMat());

auto *S22 = new mfem::UMFPackSolver();
S22->SetOperator(A22->SpMat());

// A21 must be an Operator; if you have MixedBilinearForm A21, use A21->SpMat()
LowerTriPrec P(*blockOffsets, A21->SpMat(), *S11, *S22);
/*
mfem::BlockDiagonalPreconditioner P(*blockOffsets);

UMFPackSolver *S11 = new UMFPackSolver();
S11->SetOperator(A11->SpMat());

UMFPackSolver *S22 = new UMFPackSolver();
S22->SetOperator(A22->SpMat());

P.SetDiagonalBlock(0, S11);
P.SetDiagonalBlock(1, S22);
*/

/*
mfem::BlockDiagonalPreconditioner P(*blockOffsets);

auto *P11 = new DSmoother(A11->SpMat(), 0); // or DSmoother, but you saw DS issues
auto *P22 = new DSmoother(A22->SpMat(), 0);

P.SetDiagonalBlock(0, P11);
P.SetDiagonalBlock(1, P22);
*/

//
// Solve the equations system.
//
   if(1)
   {
      FGMRESSolver solver;
      //solver.SetFlexible(true);
      solver.SetAbsTol(0);
      solver.SetRelTol(1e-6);
      solver.SetMaxIter(1000);
      solver.SetPrintLevel(1);
      solver.SetKDim(150);
      solver.SetOperator(*block);
      //solver.SetPreconditioner(P);
      solver.Mult(b, x);
      cout << solver.GetFinalRelNorm() << " GetFinalRelNorm" << endl;
   }

   GridFunction *VGF = new GridFunction(VFESpace, xV, 0);
   Glvis(mesh, VGF, "Voltage", 8, " keys 'cja'");

   GridFunction *IGF = new GridFunction(IFESpace, xI, 0);
   Glvis(mesh, IGF, "Current", 8, " keys 'cja'");


   return 0;
}



         