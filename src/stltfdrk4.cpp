//                                stltferk4
//                                
// Compile with: make stltfdrk4, need MFEM version 4.7 and GLVIS-4.3.
//
// Sample runs:  ./stltfdrk4
//
/*
Description:  stltfdrk4.cpp (single transmission line transient finite differences
runge kutta 4) simulate a single transmission line with various source
signals, source impedances and load impedances. It is based on finite difference and
Runge Kutta 4 for time stepping.

options:

*/

#include <mfem.hpp>
#include "mytools.hpp"
#include <iostream>
#include <math.h>
#include <filesystem>
#include <cstdlib>

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



Vector Zero(3);
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
   const Operator *block;
   const int size;
   
public:
   TelegrapherOperator(const Operator *block_, int size_)
      : TimeDependentOperator(size_), size(size_), block(block_)
   {}

   virtual void Mult(const Vector &x, Vector &y) const
   {
      // x is const need to copy it to then force s[0].
      Vector s(x.Size());
      s=x;
      s[0]= SourceFunction(Zero, t);

      block->Mult(s, y);
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
      double R = 220e-3;  // Resistance per unit length
      double G = 1.0e-9;  // Conductance per unit length

      double Rs = 50;
      double Rl = 50;

      double lenght = 100; //100 metre.
      int nbrSeg = 1000;  //number of segment.
      real_t h = lenght/nbrSeg;
      real_t deltaT = 0.01e-9;
      real_t endTime = 1500e-9;
      real_t Time = 0.0;


      SparseMatrix *smDi, *smDv, *smRi, *smGv, *smVs;
      Array<int> *BlockOffset;
      BlockOperator *fdBlockOperator;
      TelegrapherOperator *teleOp;

      // solution vector, represent the transmission line.
      Vector *x;
      
   public:
      TransmissionLineTransient();
      //parse the options.
      int Parser(int argc, char *argv[]);
      int CreateMatrix();
      int CreateFDBlockOperator();
      int TimeSteps();
};




TransmissionLineTransient::TransmissionLineTransient()
{
   
}
 


int TransmissionLineTransient::Parser(int argc, char *argv[])
{

   OptionsParser args(argc, argv);
   args.AddOption(&printMatrix, "-prm", "--printmatrix", "-dnprm", "--donotprintmatrix",
                  "Print of not the matrix.");
                
   args.Parse();

 //  Device device("cpu");
 //  device.Print();

   if (args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   return 0;
}

int TransmissionLineTransient::CreateMatrix()
{

   //Create the derivative matrix Dv.
   int nrow = nbrSeg;
   int ncol = nbrSeg+1;
   smDv = new SparseMatrix(nrow, ncol);
   real_t val = 1/(L*h);
   for(int i=0; i<nrow; i++)
   {
      smDv->Add(i, i+1,  val);
      smDv->Add(i,   i, -val);
   }
   

   //Create the derivative matrix Di.
   nrow = nbrSeg+1;
   ncol = nbrSeg;
   smDi = new SparseMatrix(nrow, ncol);
   val = 1/(C*h);
   for(int i=0; i<nrow; i++)
   {
      if(i<ncol) smDi->Add(i, i,    val);
      if(i-1 >= 0)   smDi->Add(i, i-1, -val);
   }
   

//Create the Ri matrix.
   nrow = nbrSeg;
   ncol = nbrSeg;
    smRi = new SparseMatrix(nrow, ncol);
    val = -R/L;
    for(int i=0; i<nrow; i++)
    {
       smRi->Add(i, i, val);
    }
    

//Create the Gv matrix.
   nrow = nbrSeg+1;
   ncol = nbrSeg+1;
   smGv = new SparseMatrix(nrow, ncol);
   val = -G/C;
   for(int i=0; i<nrow; i++)
   {
      smGv->Add(i, i, val);
   }
   

   
   //add boundary condition for Vs and Rs.
   //Vs/Rs caused current in V0.
   //make extra column for Vs.
   nrow = nbrSeg+1;
   ncol = 1;
   smVs = new SparseMatrix(nrow, ncol);
   smVs->Set(0, 0, 1/(Rs*C*h));

   // V0/Rs caused current out of V0.
   smGv->Add(0, 0, - 1/(Rs*C*h));
    
   //add boundary condition for Rl.
   // VnbrSeg/Rl caused out of VnbrSeg.
   smGv->Add(nbrSeg, nbrSeg, - 1/(Rl*C*h));

   smVs->Finalize();
   smDv->Finalize();
   smDi->Finalize();
   smRi->Finalize();
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

   {
      std::ofstream out("out/smVs.txt");
      if(printMatrix) smVs->PrintMatlab(out);
   }
   return 1;
}

int TransmissionLineTransient::CreateFDBlockOperator()
{

// 6. Define the BlockStructure
   BlockOffset = new Array<int>(4);
   (*BlockOffset)[0]=0;
   (*BlockOffset)[1]=1; 
   (*BlockOffset)[2]=nbrSeg+1; 
   (*BlockOffset)[3]=nbrSeg;
   BlockOffset->PartialSum();
   {
      std::ofstream out("out/blockOffset.txt");
      BlockOffset->Print(out, 10);
   }
  
   fdBlockOperator = new BlockOperator(*BlockOffset);
   
// Build the operator, insert each block.
  // rhsOp->SetBlock(0, 0, DofByOne);
  fdBlockOperator->SetBlock(1, 0, smVs);
  fdBlockOperator->SetBlock(1, 1, smGv);
  fdBlockOperator->SetBlock(1, 2, smDi);
  fdBlockOperator->SetBlock(2, 1, smDv);
  fdBlockOperator->SetBlock(2, 2, smRi);
   
      {
      std::ofstream out("out/fdBlockOperator.txt");
      if(printMatrix) fdBlockOperator->PrintMatlab(out);
      }
 
      assert(fdBlockOperator->Height() == 1+nbrSeg+1+nbrSeg);
      assert(fdBlockOperator->Width() ==  1+nbrSeg+1+nbrSeg);
   
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
         Vector val(x->GetData()+1, nbrSeg+1);
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

   {
      std::ofstream out("out/x.txt");
      x->Print(out, 1);
   }

   return 1;
}



int main(int argc, char *argv[])
{
   Device device("cpu");
   MemoryType mt = device.GetMemoryType();
   
   CleanOutDir();

   TransmissionLineTransient TLT;
   TLT.Parser(argc, argv);
   TLT.CreateMatrix();
   TLT.CreateFDBlockOperator();
   TLT.TimeSteps();

   return 0;
}
