//                                stlt
//                                
// Compile with: make stltfd, need MFEM version 4.7 and GLVIS-4.3.
//
// Sample runs:  ./stltfd
//
/*
Description:  stltfd (single transmission line transient finuite difference) simulate a 
single transmission line with various soure signal,
source impedance and load impedance.

*/

#include <mfem.hpp>
#include "mytools.hpp"
#include <iostream>
#include <math.h>
#include <filesystem>

using namespace std;
using namespace mfem;

class TransmissionLineTransient
{
   private:
      // 1. Parse command-line options.
      int order = 1;
      bool printMatrix = true;
      
/*
# Parameters
Lx = 10.0    # Length of the transmission line
Nx = 100     # Number of spatial points
dx = Lx / Nx  # Space step
dt = 0.01    # Time step
Nt = 200     # Number of time steps

# Circuit Parameters
L = 1.0   # Inductance per unit length
C = 1.0   # Capacitance per unit length
R = 0.1   # Resistance per unit length
G = 0.1   # Conductance per unit length
*/
// Constants for the telegrapher’s equation for RG-58, 50 ohm.
double L = 1;  // Inductance per unit length
double C = 1; // Capacitance per unit length
double R = 0.1;  // Resistance per unit length
double G = 0.1;  // Conductance per unit length
double tlLenght = 10.0; // transmission line lenght.

// source and load impedance.
double Rs = 50.0;   // source impedance.
double Rl = 50.0; //load impedance.

      real_t deltaT = 0.0001;
      real_t endTime = 2.0;
      real_t Time = 0.0;
      real_t h = 0.01; // segment lenght;

      int nbrSeg;  //number of element.

      SparseMatrix *smDv, *smDi, *smRi, *smGv;
      
      Vector *vecDer, *yn, *ynm1;
      Vector *Vn, *Vnm1, *In, *Inm1;
      
      Array<int> *BlockOffset;
      BlockMatrix *rhsBlockMatrix;
   
   public:
      TransmissionLineTransient();
      //delete all files in out dir.
      int CleanOutDir();
      //parse the options.
      int Parser(int argc, char *argv[]);
      
      int CreateMatricesVectors();
      int CreateRhsBlockOperator();
       int TimeSteps();
       int debugFileWrite();
      /*
      
      int PostPrecessing();
      int DisplayResults();
      int Save();
   */
};

real_t SourceFunction(const Vector x, real_t t)
{
   return 1.0;
   real_t y;
   if(t<3e-9) y = t/3e-9;
   else if(t>=3e-9 && t<6e-9) y = (6e-9-t)/3e-9;
   else y=0.0;

   return y *= cos(2.0 * M_PI * 1E9 * t);
}

TransmissionLineTransient::TransmissionLineTransient()
{
   nbrSeg = tlLenght / h;
   
}
 

int TransmissionLineTransient::CleanOutDir()
{
    system("rm -f out/*");
    return 1;
}

 


int TransmissionLineTransient::Parser(int argc, char *argv[])
{

   OptionsParser args(argc, argv);
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


int TransmissionLineTransient::CreateMatricesVectors()
{

   //Create the derivative matrix.
   smDv = new SparseMatrix(nbrSeg+1);
   real_t val = -deltaT/(L*2.0*h);
   for(int i=0; i<nbrSeg+1; i++)
   {
      if(i+1<nbrSeg+1) smDv->Add(i, i+1, val);
      if(i-1>= 0) smDv->Add(i, i-1, -val);
   }
   smDv->Finalize();

   //Create the derivative matrix.
   smDi = new SparseMatrix(nbrSeg+1);
   val = -deltaT/(C*2.0*h);
   for(int i=0; i<nbrSeg+1; i++)
   {
      if(i+1<nbrSeg+1) smDi->Add(i, i+1, val);
      if(i-1>= 0) smDi->Add(i, i-1, -val);
   }
   smDi->Finalize();

  
   
   /*
//Create the derivative matrix.
    smDv = new SparseMatrix(nbrSeg+1);
    real_t val = -deltaT/(L*h);
    for(int i=1; i<nbrSeg+1; i++)
    {
       smDv->Add(i, i, val);
       smDv->Add(i, i-1, -val);
    }
    smDv->Add(0, 1, val);
    smDv->Add(0, 0, -val);
    smDv->Finalize();


//Create the derivative matrix.
    smDi = new SparseMatrix(nbrSeg+1);
    val = -deltaT/(C*h);
    for(int i=1; i<nbrSeg+1; i++)
    {
       smDi->Add(i, i, val);
       smDi->Add(i, i-1, -val);
    }
    smDi->Add(0, 1, val);
    smDi->Add(0, 0, -val);
    smDi->Finalize();

    */
//Create the Ri matrix.
    smRi = new SparseMatrix(nbrSeg+1);
    for(int i=0; i<nbrSeg+1; i++)
    {
       smRi->Add(i, i, -deltaT*R/L);
    }
    smRi->Finalize();

//Create the Gv matrix.
   smGv = new SparseMatrix(nbrSeg+1);
   for(int i=0; i<nbrSeg+1; i++)
   {
      smGv->Add(i, i, -deltaT*G/C);
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

   vecDer = new Vector(2*(nbrSeg+1));
   yn = new Vector(2*(nbrSeg+1));   
   ynm1 = new Vector(2*(nbrSeg+1));

   In = new Vector(nbrSeg+1);   
   Inm1 = new Vector(nbrSeg+1);

   Vn = new Vector(nbrSeg+1);   
   Vnm1 = new Vector(nbrSeg+1);

   *Vn=0.0;  *Vnm1=0.0; 
   *In=0.0;  *Inm1=0.0; 
   

   return 1;
}



int TransmissionLineTransient::CreateRhsBlockOperator()
{

// 6. Define the BlockStructure of lhsMatrix
   BlockOffset = new Array<int>(3);
   (*BlockOffset)[0]=0;
   (*BlockOffset)[1]=nbrSeg+1; 
   (*BlockOffset)[2]=nbrSeg+1; 
   BlockOffset->PartialSum();
   {
      std::ofstream out("out/blockOffset.txt");
      BlockOffset->Print(out, 10);
   }
  
   Device device("cpu");
   MemoryType mt = device.GetMemoryType();

   rhsBlockMatrix = new BlockMatrix(*BlockOffset);
   
// Build the operator, insert each block.
// row 0 ...
   
   rhsBlockMatrix->SetBlock(0, 0, smGv);
   rhsBlockMatrix->SetBlock(0, 1, smDi);
   rhsBlockMatrix->SetBlock(1, 0, smDv);
   rhsBlockMatrix->SetBlock(1, 1, smRi);
   rhsBlockMatrix->Finalize(); 

      {
      std::ofstream out("out/rhsBlockMatrix.txt");
      if(printMatrix) rhsBlockMatrix->PrintMatlab(out);
      }

      assert(rhsBlockMatrix->Height() == 2*(nbrSeg+1));
      assert(rhsBlockMatrix->Width() == 2*(nbrSeg+1));

   return 1;
}



int TransmissionLineTransient::TimeSteps()
{
   cout << deltaT << " deltaT\n";
   cout << h/sqrt(L*C) << " h/sqrt(L*C)\n";
   
   FunctionCoefficient SourceCoefficient(SourceFunction);
   
   *ynm1 = 0.0;

   if(1)
   {
      for(int i=0; i<nbrSeg+1; i++)
      {
         (*ynm1)[i] = sin(2.0 * M_PI * i/(nbrSeg+1)/0.5);
         (*ynm1)[i+nbrSeg+1] =0.5 * sin(2.0 * M_PI * i/(nbrSeg+1)/0.5);
      }
   }

   *yn = 0.0;
   
   {
      std::ofstream out("out/ynm1.txt");
      if(printMatrix) ynm1->Print(out, 1);
   }

   // x, position for the SourceFunction.
   Vector x(1); x[0]=0;
   Vector srcOutput(nbrSeg); srcOutput=0.0;
   Time=0.0;
   int i=0;
   (*Vnm1)[nbrSeg/2]=1.0;
   debugFileWrite();
   while(Time<endTime)
   {
      //  (*ynm1)[nbrSeg/2] = SourceFunction(x, Time);
      // srcOutput[i]= (*ynm1)[5000];
      //  rhsBlockMatrix->AddMult(*ynm1, *yn);
 
   for(int j=0;j<nbrSeg;j++)
   {
      (*In)[j] = (*Inm1)[j] - deltaT/L * (((*Vnm1)[j+1]-(*Vnm1)[j])/h + R * (*Inm1)[j]);
      (*Vn)[j] = (*Vnm1)[j] - deltaT/C * (((*Inm1)[j+1]-(*Inm1)[j])/h + G * (*Vnm1)[j]);
   }

 // Matrix-Vector Multiplication y = y + val*A*x
 //  virtual void AddMult(const Vector & x, Vector & y, const real_t val = 1.) const;
 
   cout << In->Norml2() << " In Norm L2\n";
   cout << Vn->Norml2() << " Vn Norm L2\n";

   debugFileWrite();
   *Inm1 = *In;
   *Vnm1 = *Vn;

   if(0)
   {
      std::ofstream out("out/srcoutput.txt");
      if(printMatrix) srcOutput.Print(out, 1);
   }

   Time += deltaT;
   i++;

   }

   return 1;
}

int TransmissionLineTransient::debugFileWrite()
{
   {
     std::ofstream out("out/In.txt");
     if(printMatrix) In->Print(out, 1);
   }

   {
   std::ofstream out("out/Inm1.txt");
   if(printMatrix) Inm1->Print(out, 1);
   }

   {
      std::ofstream out("out/Vn.txt");
      if(printMatrix) Vn->Print(out, 1);
    }
 
    {
    std::ofstream out("out/Vnm1.txt");
    if(printMatrix) Vnm1->Print(out, 1);
    }

   return 1;
}


int main(int argc, char *argv[])
{

   TransmissionLineTransient TLT;

   TLT.Parser(argc, argv);
   TLT.CleanOutDir();
   TLT.CreateMatricesVectors();
   TLT.CreateRhsBlockOperator();
   TLT.TimeSteps();
   /*
   TLT.CreateEssentialBoundary();
   

   TLT.CreatexVector();
   TLT.CreatePreconditionner();
   TLT.Solver();
   TLT.PostPrecessing();
   TLT.DisplayResults();
   TLT.Save();
   */
   return 0;
}
