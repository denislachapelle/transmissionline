//                                stltferk4
//                                
// Compile with: make stltfdrk4, need MFEM version 4.7 and GLVIS-4.3.
//
// Sample runs:  ./stltfdrk4
//
/*
Description:  stltfe (single transmission line transient finite differences
runge kutta 4) simulate a single transmission line with various source
signals, source impedances and load impedances.

options:

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

// TelegrapherOperator is used to timestep.
class TelegrapherOperator : public TimeDependentOperator
{
private:
   const Operator *block;
   const int size;
   
public:
   TelegrapherOperator(const Operator *block_, int size_)
      : TimeDependentOperator(size_), size(size_), block(block_)
   {}

   virtual void Mult(const Vector &x, Vector &y) const
   {
      (*block).Mult(x, y);
   }
};
 

class TransmissionLineTransient
{
   private:
      // 1. Parse command-line options.
      bool printMatrix = true;
            
      // Constants for the telegrapher’s equation for RG-58, 50 ohm.
      double L = 250e-9;  // Inductance per unit length
      double C = 100.0e-12; // Capacitance per unit length
      double R = 1000e-3;  // Resistance per unit length
      double G = 1.0e-9;  // Conductance per unit length

      // source and load impedance.
      double Rs = 50.0;   // source impedance.
      double Rl = 50.0; //load impedance.

      double lenght = 100;
      int nbrSeg = 1000;  //number of segment.
      real_t h = lenght/nbrSeg;
      real_t deltaT = 0.01e-9;
      real_t endTime = 200e-9;
      real_t Time = 0.0;
      SparseMatrix *smDi, *smDv, *smRi, *smGv;
      BlockOperator *fdBlockOperator;
      Array<int> *RowBlockOffset;
      Array<int> *ColBlockOffset;
      TelegrapherOperator *teleOp;
      Vector *x;
      
   public:
      TransmissionLineTransient();
      //delete all files in out dir.
      int CleanOutDir();
      //parse the options.
      int Parser(int argc, char *argv[]);
      int createFDBlockOperator();
      int CreateFiniteDiffMatrix();
      int TimeSteps();
      int debugFileWrite();
      int TestSourceFunction();
};
/*
generate a time dependent signal
*/
real_t SourceFunction(const Vector x, real_t t)
{
   if(0) 
   {
      /* 1GHz sinewave multiply by triangular wave of 20ns.*/
      real_t th = 10e-9;
      if(t<2*th) return 4.0 * t/th*(1-t/th) * sin(2*M_PI*1e9*t);
      else return 0.0;
   }
   else if(1)
   {
      /* gaussian pulse of tw wide centered at tc.*/

      real_t tw = 20e-9;
      real_t tc = 100e-9;
      if(t<2*tc) return 4.0 * t/(2.0*tc)*(1-t/(2.0*tc)) * exp(-pow(((t-tc)/tw), 2.0));
      else return 0.0;
   }
}

int TransmissionLineTransient::TestSourceFunction()
{
   double t=0.0, dt=0.1e-9, endTime=200e-9;
   int size=endTime/dt;
   Vector x(3);
   Vector y(size);
   for(int i=0; i<size; i++)
   {
      y(i)=SourceFunction(x, t);
      t+=dt;
   }



   return 0;
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

int TransmissionLineTransient::CreateFiniteDiffMatrix()
{

   //Create the derivative matrix Dv.
   smDv = new SparseMatrix(nbrSeg+1);
   real_t val = -1/(L*2.0*h);
   for(int i=0; i<nbrSeg+1; i++)
   {
      if(i+1<nbrSeg+1) smDv->Add(i, i+1, val);
      if(i-1>= 0) smDv->Add(i, i-1, -val);
   }
   smDv->Finalize();

   //Create the derivative matrix Di.
   smDi = new SparseMatrix(nbrSeg+1);
   val = -1/(C*2.0*h);
   for(int i=0; i<nbrSeg+1; i++)
   {
      if(i+1<nbrSeg+1) smDi->Add(i, i+1, val);
      if(i-1>= 0) smDi->Add(i, i-1, -val);
   }
   smDi->Finalize();

//Create the Ri matrix.
    smRi = new SparseMatrix(nbrSeg+1);
    for(int i=0; i<nbrSeg+1; i++)
    {
       smRi->Add(i, i, -R/L);
    }
    smRi->Finalize();

//Create the Gv matrix.
   smGv = new SparseMatrix(nbrSeg+1);
   for(int i=0; i<nbrSeg+1; i++)
   {
      smGv->Add(i, i, -G/C);
   }
   smGv->Finalize();

   cout << smDi->Height() << " smDi->Height()\n";
   cout << smDi->Width() << " smDi->Width()\n";
   cout << smDv->Height() << " smDv->Height()\n";
   cout << smDv->Width() << "smDv->Width()\n";
   cout << smRi->Height() << " smRi->Height()\n";
   cout << smRi->Width() << " smRi->Width()\n";
   cout << smRi->Height() << " smRi->Height()\n";
   cout << smRi->Width() << " smRi->Width()\n";
   

   {
      std::ofstream out("out/smDv.txt");
      if(printMatrix) smDv->PrintMatlab(out);
   }

   {
      std::ofstream out("out/smDi.txt");
      if(printMatrix) smDi->PrintMatlab(out);
   }

   {
      std::ofstream out("out/smRi.txt");
      if(printMatrix) smRi->PrintMatlab(out);
   }

   {
      std::ofstream out("out/smGv.txt");
      if(printMatrix) smGv->PrintMatlab(out);
   }
   return 1;
}

int TransmissionLineTransient::createFDBlockOperator()
{

// 6. Define the BlockStructure of lhsMatrix
   RowBlockOffset = new Array<int>(3);
   (*RowBlockOffset)[0]=0;
   (*RowBlockOffset)[1]=nbrSeg+1; 
   (*RowBlockOffset)[2]=nbrSeg+1; 
   RowBlockOffset->PartialSum();
   {
      std::ofstream out("out/rowblockOffset.txt");
      RowBlockOffset->Print(out, 10);
   }
  
   ColBlockOffset = new Array<int>(3);
   (*ColBlockOffset)[0]=0;
   (*ColBlockOffset)[1]=nbrSeg+1; 
   (*ColBlockOffset)[2]=nbrSeg+1; 
   ColBlockOffset->PartialSum();
   {
      std::ofstream out("out/colblockOffset.txt");
      ColBlockOffset->Print(out, 10);
   }
  
   Device device("cpu");
   MemoryType mt = device.GetMemoryType();

   fdBlockOperator = new BlockOperator(*RowBlockOffset, *ColBlockOffset);
   
// Build the operator, insert each block.
  // rhsOp->SetBlock(0, 0, DofByOne);
  fdBlockOperator->SetBlock(0, 0, smGv);
  fdBlockOperator->SetBlock(0, 1, smDi);
  fdBlockOperator->SetBlock(1, 0, smDv);
  fdBlockOperator->SetBlock(1, 1, smRi);
   

      {
      std::ofstream out("out/fdBlockOperator.txt");
      if(printMatrix) fdBlockOperator->PrintMatlab(out);
      }
 
      assert(fdBlockOperator->Height() == 2 * (nbrSeg+1));
      assert(fdBlockOperator->Width() == 2 * (nbrSeg+1));
   
   return 1;
}



int TransmissionLineTransient::TimeSteps()
{
   cout << deltaT << " deltaT\n";
   cout << sqrt(L*C) << " sqrt(L*C)\n";

   teleOp = new TelegrapherOperator(fdBlockOperator, 2 * (nbrSeg+1));
   
   RK4Solver solver;
   solver.Init(*teleOp);

   x = new Vector(2*(nbrSeg+1));
   *x = 0.0;

   Vector Zero(3);

   while(Time<endTime)
   {
      x[0]=SourceFunction(Zero, Time);
      solver.Step(*x, Time, deltaT);

      if((int)(Time/deltaT) % 1000 == 0)
      {
         std::ofstream out("out/x.txt");
         x->Print(out, 1);
      }

   Time += deltaT;
   }

   {
      std::ofstream out("out/x.txt");
      x->Print(out, 1);
   }

   return 1;
}


int TransmissionLineTransient::debugFileWrite()
{

   return 1;


}

int main(int argc, char *argv[])
{

   TransmissionLineTransient TLT;

   TLT.Parser(argc, argv);
   TLT.CleanOutDir();
   TLT.CreateFiniteDiffMatrix();
   TLT.createFDBlockOperator();
   TLT.TimeSteps();
   TLT.TestSourceFunction();

   return 0;
}
