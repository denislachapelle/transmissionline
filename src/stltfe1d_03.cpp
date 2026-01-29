//                                stltfe1d_03
//                                
// Compile with: make stltferk4, need MFEM version 4.8 and GLVIS-4.3.
//
// Sample runs:  ./stltfe1d_03
//
/*
Description:  stltfe (single transmission line transient finite element 1D) 
simulate a single transmission line with various source
signals, source impedances and load impedances.

For mor details see stltfe1d.pdf (latex).

January 2026: did not work yet, just begin.
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
real_t deltaT = 0.5e-9;
real_t endTime = 1000e-9;

real_t SourceFunctionGaussianPulse(const Vector x, real_t t)
{
   /* gaussian pulse of tw wide centered at tc.*/
   real_t tw = 100e-9;
   real_t tc = 300e-9;
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
   return SourceFunctionGaussianPulse(x, t);
}


typedef double (*SourceFunctionType)(const Vector x, double time);

class VsRsCoefficient : public Coefficient
{
   private:
      Vector zeroVec;
      double t, Rs;
      SourceFunctionType vs;
      
   public:
      VsRsCoefficient(double Rs_, SourceFunctionType vs_)
      : Rs(Rs_), vs(vs_)
      {
         zeroVec.SetSize(3);
         zeroVec=0.0;
      }

      void SetTime(real_t t_) {t = t_;}

    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {

        return vs(zeroVec, t) / Rs;  // forcing term
    }
};





//
// TelegrapherOperator is used to timestep, it compute the d(.)/dt.
//
class TelegrapherOperator : public TimeDependentOperator
{
private:
   
   FiniteElementSpace *FESpace;
   BilinearForm *A11, *A22;
   BilinearForm *B11, *B12, *B22;
   MixedBilinearForm  *B21;
   LinearForm *C1;
   Array<int> *boundary_dofs;
   Array<int> *inputBdrMarker, *outputBdrMarker;
   VsRsCoefficient *VsRs; 
   LinearForm *LFVS;

   BlockOperator *lhsOp;
   Array<int> *lhsBlockOffset;

   BlockDiagonalPreconditioner *prec;
            
   BlockOperator *rhsOp;
   Array<int> *rhsBlockOffset;

   // Constants for the telegrapher’s equation for RG-58, 50 ohm.
   double L = 250e-9;  // Inductance per unit length
   double C = 100e-12; // Capacitance per unit length
   double R = 220e-3;  // Resistance per unit length
   double G = 1.0e-9;  // Conductance per unit length

   // source and load impedance.
   double Rs = 50.0;   // source impedance.
   double Rl = 50.0; //load impedance.

public:
   TelegrapherOperator(FiniteElementSpace *FESpace_)
      : TimeDependentOperator(2*FESpace_->GetVSize(), 0.0), FESpace(FESpace_)
   {

      //
      // input port boundary marker.
      //
      inputBdrMarker = new Array<int>(FESpace->GetMesh()->bdr_attributes.Size());
      assert(inputBdrMarker->Size() == 2);
      *inputBdrMarker = 0;
      (*inputBdrMarker)[0] = 1; 
     
      //
      // output port boundary marker.
      //
      outputBdrMarker = new Array<int>(FESpace->GetMesh()->bdr_attributes.Size());
      assert(outputBdrMarker->Size() == 2);
      *outputBdrMarker = 0;
      (*outputBdrMarker)[1] = 1; 

      //
      // Define the mass and stiffnes matrix
      //
      ConstantCoefficient one(1.0), mOne(-1.0);

      
      A11 = new BilinearForm(FESpace);
      ConstantCoefficient A11Coeff(L/deltaT);
      A11->AddDomainIntegrator(new MassIntegrator(A11Coeff));
      A11->Assemble();
      A11->Finalize();

      
      A22 = new BilinearForm(FESpace);
      ConstantCoefficient A22Coeff(C/deltaT);
      A22->AddDomainIntegrator(new MassIntegrator(A22Coeff));
      A22->Assemble();
      A22->Finalize();


      B11 = new BilinearForm(FESpace);
      ConstantCoefficient B11Coeff(L/deltaT - R);
      B11->AddDomainIntegrator(new MassIntegrator(B11Coeff));
      B11->Assemble();
      B11->Finalize();

      B12 = new BilinearForm(FESpace);
      B12->AddDomainIntegrator(new DerivativeIntegrator(mOne, 0));
      B12->Assemble();
      B12->Finalize();

      B21 = new MixedBilinearForm(FESpace, FESpace);      
      B21->AddDomainIntegrator(new MixedScalarWeakDerivativeIntegrator(mOne));
      B21->Assemble();
      B21->Finalize();

      B22 = new BilinearForm(FESpace);
      ConstantCoefficient B22Coeff(C/deltaT - G);
      B22->AddDomainIntegrator(new MassIntegrator(B22Coeff));
      ConstantCoefficient mOneOverRsCoeff(-1.0/Rs);
      ConstantCoefficient mOneOverRlCoeff(-1.0/Rl);
      B22->AddBoundaryIntegrator(new BoundaryMassIntegrator(mOneOverRsCoeff), *inputBdrMarker);
      B22->AddBoundaryIntegrator(new BoundaryMassIntegrator(mOneOverRlCoeff), *outputBdrMarker);
      B22->Assemble();
      B22->Finalize();

      //
      // define the linear form injecting the source signal.
      //
     
      VsRs = new VsRsCoefficient(Rs, SourceFunction);
      VsRs->SetTime(0.0);
      
      LFVS = new LinearForm(FESpace);
      LFVS->AddBoundaryIntegrator(new BoundaryLFIntegrator(*VsRs), *inputBdrMarker);
      LFVS->Assemble();
      
      MyPrintFile(LFVS, printMatrix, "out/LFVS.txt");

      
      // 6. Define the BlockStructure of lhs and Matrice.
      Array<int> *blockOffsets = new Array<int>(3);
      (*blockOffsets)[0]=0;
      (*blockOffsets)[1]=A11->Height(); 
      (*blockOffsets)[2]=A22->Height(); 
      blockOffsets->PartialSum();
      MyPrintFile(blockOffsets, printMatrix, "out/lhsRowBlockOffset.txt" );
   
      
      lhsOp = new BlockOperator(*blockOffsets);
      
   // Build the operator, insert each block.
   // row 0 ...
         
      lhsOp->SetBlock(0, 0, A11, 1.0);
      lhsOp->SetBlock(1, 1, A22, 1.0);

      {
      std::ofstream out("out/lhsOp.txt");
      if(printMatrix) lhsOp->PrintMatlab(out);
      }

      rhsOp = new BlockOperator(*blockOffsets);
      
   // Build the operator, insert each block.
   // rhsOp->SetBlock(0, 0, DofByOne);
      rhsOp->SetBlock(0, 0, B11);
      rhsOp->SetBlock(0, 1, B12);
      rhsOp->SetBlock(1, 0, B21);
      rhsOp->SetBlock(1, 1, B22);

      {
      std::ofstream out("out/rhsOp1.txt");
      if(printMatrix) rhsOp->PrintMatlab(out);
      }

      prec = new BlockDiagonalPreconditioner(*blockOffsets);

      // Example: setting Jacobi or AMG on each block
      prec->SetDiagonalBlock(0, new GSSmoother(A11->SpMat()));
      prec->SetDiagonalBlock(1, new GSSmoother(A22->SpMat()));


   }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      Vector b(x.Size()); b=0.0;
      rhsOp->Mult(x, b);

      {
         std::ofstream out("out/xrhs.txt");
         if(printMatrix) x.Print(out, 1);
      }  

      {
         std::ofstream out("out/brhs.txt");
         if(printMatrix) b.Print(out, 1);
      }  

      
      //update LinearForm LFVS.
      VsRs->SetTime(GetTime());
      //*LFVS=0;
      LFVS->Assemble();

// Slice the second block (starts at nbrDof)
   int nbrDof = FESpace->GetVSize();
   Vector b_V(b, nbrDof, nbrDof);
   b_V += *LFVS; // This is the B^a * Vs(t)/Rs term
      

      //solve for y.
      GMRESSolver solver;
      solver.SetAbsTol(1e-15);
      solver.SetRelTol(1e-12);
      solver.SetMaxIter(200);
      solver.SetPrintLevel(-1);
      //solver.SetKDim(100);
      solver.SetOperator(*lhsOp);
      //solver.SetPreconditioner(*prec);
      solver.Mult(b, y);


   }
};
 





int main(int argc, char *argv[])
{

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, occ::, RAJA and OpenMP based on command line options.
   Device device("cpu");
   device.Print();
   MemoryType mt = device.GetMemoryType();

   //
   // CParse the command arguments.
   //  
   int order = 1; 
   int nbrSeg = 100;
   real_t lenght = 100.0;
   
   



   OptionsParser args(argc, argv);
   args.AddOption(&nbrSeg, "-ns", "--nbrSeg",
                  "Number of segment in the transmission line.");
   args.AddOption(&lenght, "-l", "--lenght",
                  "Lenght of the transmission line.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&deltaT, "-dt", "--deltaT",
                  "DeltaT.");
   args.AddOption(&endTime, "-et", "--endTime",
                  "endTime: Simulation time.");
   args.AddOption(&printMatrix, "-prm", "--printmatrix", "-dnprm", "--donotprintmatrix",
                  "Print of not the matrix.");
                 
   args.Parse();

   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   //
   // Clean the output directory.
   //   
    system("rm -f out/*");
    

   //
   // Create the transmission line mesh.
   //   
   // Creates 1D mesh, divided into n equal intervals.
   Mesh *mesh;
   mesh = new Mesh();
   *mesh = mfem::Mesh::MakeCartesian1D(nbrSeg, lenght); 
   assert(mesh->Dimension()==1); //this software works only for dimension 1.
   mesh->PrintInfo();
   MyPrintFile(mesh, printMatrix, "out/meshprint.txt");
   
   //
   // Create the finite element space,
   //
   H1_FECollection *FEC;
   FEC = new H1_FECollection(order, 1);
   FiniteElementSpace *FESpace;
   FESpace = new FiniteElementSpace(mesh, FEC);
   int nbrDof = FESpace->GetTrueVSize(); 
   cout << nbrDof << " degree of freedom\n";   

   //
   //prepare the x vector [I V]^T.
   //
   Vector x(2*nbrDof);
   x = 0.0;

   //
   // Time step loops
   //
   TelegrapherOperator *teleOp;
   teleOp = new TelegrapherOperator(FESpace);
      
   Vector sourceFunctionVector(endTime/deltaT + 1);
   sourceFunctionVector=0.0;
   int sourceFunctionCounter = 0;
   Vector Zero(3);
   Zero=0.0;

   int plotCount = 0;
   int nbrOfPlot = 5;
   double plotTimeInterval = endTime/nbrOfPlot;
   real_t time = 0.0;

   while(1)
   {
      cout << time << endl;

      
      //Save in a Vector for testing SourceFunction.
     // sourceFunctionVector[sourceFunctionCounter++] = SourceFunction(Zero, time);  

      //this section of code was quite tested.

      
      teleOp->SetTime(time);
      
      Vector xnew(x.Size()); xnew=0.0;
      teleOp->Mult(x, xnew);
      x = xnew;  

      GridFunction *VGF, *IGF;

      if(time/plotTimeInterval >= 1.0*plotCount)
      {
         VGF = new GridFunction(FESpace);
         IGF = new GridFunction(FESpace);
  
         // rebuild GFR and GFI from x.
         IGF->MakeRef(FESpace, x, 0);
         VGF->MakeRef(FESpace, x, nbrDof);

         std::string s = "V after " + std::to_string(time * 1e9) + "ns." ;
         Glvis(mesh, VGF, s, 8, "keys 'caRR'" );
      //   s = "I" + std::to_string(plotCount);
       //  Glvis(mesh, IGF, s, 8, "keys 'caRR'" );

         delete VGF;
         delete IGF;

         plotCount++;
         if (time>endTime) break;
         cout << "pause" << endl; getchar();
      }
      
      time += deltaT;
      }
 
   {
      std::ofstream out("out/sourcefunction.txt");
      if(printMatrix) sourceFunctionVector.Print(out, 1);
   }
   
   return 0;
}
