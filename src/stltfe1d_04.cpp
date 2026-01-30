//                                stltfe1d_04
//                                
// Compile with: make stltfe1d_04, need MFEM version 4.8 and GLVIS-4.2.
//
// Sample runs:  ./stltfe1d_03
//
/*
Description:  stltfe1d (single transmission line transient finite element 1D) 
simulate a single transmission line with various source
signals, source impedances and load impedances.

The time stepping is backward  euler.

For mor details see stltfe1d.pdf (latex).


January 21, 2026: The software work fine.
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
real_t deltaT = 0.05e-9;  //time stepping delta.
real_t endTime = 1000e-9;

int order = 1; 
int nbrSeg = 100;
real_t lenght = 100.0;
   
 
// Constants for the telegrapher’s equation for RG-58, 50 ohm.
real_t L = 250e-9;  // Inductance per unit length
real_t C = 100e-12; // Capacitance per unit length
real_t R = 220e-3;  // Resistance per unit length
real_t G = 1.0e-9;  // Conductance per unit length

// source and load impedance.
real_t Rs = 50.0;   // source impedance.
real_t Rl = 50.0; //load impedance.

int functionIndex=0;

int vis = 2; //gnuplot.

real_t fp1=NAN, fp2=NAN;

#include "mfem.hpp"
#include <fstream>

using namespace mfem;

void ExportForGnuplot(GridFunction &u, int ref_points, std::string &out) {

    Mesh *mesh = u.FESpace()->GetMesh();
   std::ostringstream ssout;
    // Integration rules can give us a uniform lattice of points
    // Or we can manually loop through the reference space
    for (int i = 0; i < mesh->GetNE(); i++) {
        ElementTransformation *T = mesh->GetElementTransformation(i);
        
        for (int j = 0; j < ref_points; j++) {
            // Create a local coordinate xi from -1 to 1
            double xi_val = -1.0 + 2.0 * j / (ref_points - 1);
            IntegrationPoint ip;
            ip.x = xi_val; // For 1D; for 2D use ip.y as well

            Vector phys_x;
            T->Transform(ip, phys_x); // Get physical coordinates
            
            double val = u.GetValue(i, ip); // Get the H2 interpolated value
            
            ssout << phys_x(0) << " " << val << "\n";
        }
        // Separate elements by a blank line for Gnuplot "with lines"
        ssout << "\n"; 
    }
    out = ssout.str();
}


FILE* CreatePlot5x2()
{
    FILE *gnuplotPipe = popen("gnuplot -persistent", "w");
    if (gnuplotPipe) {
      fprintf(gnuplotPipe, "set terminal qt size 1000,800 noraise title 'MFEM Simulation: V left, I right'\n");
        fprintf(gnuplotPipe, "set multiplot layout 5,2\n");
        
        // GLOBAL SETTINGS to save space
        fprintf(gnuplotPipe, "unset key\n");            // Fixes the "plot titles into key" warning
        fprintf(gnuplotPipe, "set tmargin 2\n");        // Tighten top margin
        fprintf(gnuplotPipe, "set bmargin 2\n");        // Tighten bottom margin
        
        // FIX SCALING (Prevents the "thick line" noise issue)
        // Adjust these numbers to match your expected source voltage/current
        //fprintf(gnuplotPipe, "set yrange [-1.2:1.2]\n"); 
    }
    return gnuplotPipe;
}

void PlotRow(FILE* gnuplotPipe, Mesh *mesh, std::string &VS, std::string &IS, double time)
{
    if (gnuplotPipe) {
        // --- Left Plot: Voltage ---
        //fprintf(gnuplotPipe, "set title 'V: %.1f ns' font ',8'\n", time * 1e9);
        
        fprintf(gnuplotPipe, "plot '-' with lines lw 1.5 lc 'blue'\n");
        fprintf(gnuplotPipe, "%s", VS.c_str());
        /*
        for (int i = 0; i < VGF->Size(); i++) {
            real_t pos;
            mesh->GetNode(i, &pos);
            fprintf(gnuplotPipe, "%f %f\n", pos, (*VGF)(i));
        }
        */
        fprintf(gnuplotPipe, "e\n"); 
        
        // --- Right Plot: Current ---
        // We set a different range for current if it's much smaller than voltage
        //fprintf(gnuplotPipe, "set yrange [-0.05:0.05]\n"); 
        //fprintf(gnuplotPipe, "set title 'I: %.1f ns' font ',8'\n", time * 1e9);
        fprintf(gnuplotPipe, "plot '-' with lines lw 1.5 lc 'red'\n");
        fprintf(gnuplotPipe, "%s", IS.c_str());
        /*
        for (int i = 0; i < IGF->Size(); i++) {
            real_t pos;
            mesh->GetNode(i, &pos);
            fprintf(gnuplotPipe, "%f %f\n", pos, (*IGF)(i));
        }
        */
        fprintf(gnuplotPipe, "e\n"); 
        
        // Reset yrange for the next row's Voltage plot
        //fprintf(gnuplotPipe, "set yrange [-1.2:1.2]\n");

        fflush(gnuplotPipe);
    }
}

void ClosePlot5x2(FILE* gnuplotPipe)
{
    if (gnuplotPipe) {
        fprintf(gnuplotPipe, "unset multiplot\n");
        pclose(gnuplotPipe); // Properly close the pipe
    }
}


real_t SourceFunctionGaussianPulse(const Vector x, real_t t)
{
   /* gaussian pulse of tw wide centered at tc.*/
   real_t tw = (std::isnan(fp1)) ? 100e-9 : fp1;
   real_t tc = (std::isnan(fp2)) ? 300e-9 : fp2;
   if(t<2*tc) return 4.0 * t/(2.0*tc)*(1-t/(2.0*tc)) * exp(-pow(((t-tc)/tw), 2.0));
   else return 0.0;
}

real_t SourceFunctionStep(const Vector x, real_t t)
{
      //step.
      real_t tau = (std::isnan(fp1)) ? 5e-9 : fp1; //time constant.
      return 1.0 - exp(-t/tau);
}

real_t SourceFunctionSine(const Vector x, real_t t)
{
   real_t freq = (std::isnan(fp1)) ? 10e6 : fp1;
   return sin(2*M_PI*freq*t);
}


/*
generate a time dependent signal
*/
real_t SourceFunction(const Vector x, real_t t)
{   
   switch(functionIndex) 
   {
      case 0:
      return SourceFunctionGaussianPulse(x, t);
      case 1:
      return SourceFunctionStep(x, t);
      case 2:
      return SourceFunctionSine(x, t);
      MFEM_VERIFY(false, "functionIndex out of range");
   }
   return 0;
}


typedef double (*SourceFunctionType)(const Vector x, double time);

class VsRsCoefficient : public Coefficient
{
   private:
      Vector zeroVec;
      double t, k;
      SourceFunctionType vs;
      
   public:
      VsRsCoefficient(double k_, SourceFunctionType vs_)
      : k(k_), vs(vs_)
      {
         zeroVec.SetSize(3);
         zeroVec=0.0;
      }

      void SetTime(real_t t_) {t = t_;}

    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
      return k * vs(zeroVec, t);  // forcing term
    }
};

//
// TelegrapherOperator is used to timestep, it compute X(n+1.
//
class TelegrapherOperator : public TimeDependentOperator
{
private:
   
   FiniteElementSpace *FESpace;
   BilinearForm *A11, *A22;
   BilinearForm *B11, *B12, *B22;
   MixedBilinearForm  *B21;
   LinearForm *C2;
   Array<int> *inputBdrMarker, *outputBdrMarker;
   VsRsCoefficient *VsRs; 

   Array<int> *blockOffsets;
   BlockOperator *lhsOp;
   BlockOperator *rhsOp;
   BlockDiagonalPreconditioner *prec;
  
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
      ConstantCoefficient B11Coeff(L/deltaT + R);
      B11->AddDomainIntegrator(new MassIntegrator(B11Coeff));
      B11->Assemble();
      B11->Finalize();

      B12 = new BilinearForm(FESpace);
      B12->AddDomainIntegrator(new DerivativeIntegrator(mOne, 0));
      B12->Assemble();
      B12->Finalize();

      B21 = new MixedBilinearForm(FESpace, FESpace);      
      B21->AddDomainIntegrator(new MixedScalarWeakDerivativeIntegrator(mOne)); // weak derivative is (-lU, dv/dx)
      B21->Assemble();
      B21->Finalize();

      B22 = new BilinearForm(FESpace);
      ConstantCoefficient B22Coeff(C/deltaT + G);
      B22->AddDomainIntegrator(new MassIntegrator(B22Coeff));
      ConstantCoefficient oneOverRsCoeff(1.0/Rs);
      ConstantCoefficient oneOverRlCoeff(1.0/Rl);
      B22->AddBoundaryIntegrator(new BoundaryMassIntegrator(oneOverRsCoeff), *inputBdrMarker);
      B22->AddBoundaryIntegrator(new BoundaryMassIntegrator(oneOverRlCoeff), *outputBdrMarker);
      B22->Assemble();
      B22->Finalize();

      //
      // define the linear form injecting the source signal.
      //
     
      VsRs = new VsRsCoefficient(1.0 / (Rs), SourceFunction);
      VsRs->SetTime(0.0);
      
      C2 = new LinearForm(FESpace);
      C2->AddBoundaryIntegrator(new BoundaryLFIntegrator(*VsRs), *inputBdrMarker);
      C2->Assemble();
      MyPrintFile(C2, printMatrix, "out/C2.txt");
      
      // 6. Define the BlockStructure of lhs and Matrice.
      blockOffsets = new Array<int>(3);
      (*blockOffsets)[0]=0;
      (*blockOffsets)[1]=A11->Height(); 
      (*blockOffsets)[2]=A22->Height(); 
      blockOffsets->PartialSum();
      MyPrintFile(blockOffsets, printMatrix, "out/blockOffsets.txt" );
   
      
    
      
   // Build the operator, insert each block.
   // row 0 ...
      lhsOp = new BlockOperator(*blockOffsets);
      lhsOp->SetBlock(0, 0, B11);
      lhsOp->SetBlock(0, 1, B12);
      lhsOp->SetBlock(1, 0, B21);
      lhsOp->SetBlock(1, 1, B22);
      MyPrintFile(lhsOp, printMatrix, "out/lhsOp.txt" );
   
      
   // Build the operator, insert each block.
   // rhsOp->SetBlock(0, 0, DofByOne);
      rhsOp = new BlockOperator(*blockOffsets);
      rhsOp->SetBlock(0, 0, A11, 1.0);
      rhsOp->SetBlock(1, 1, A22, 1.0);
      MyPrintFile(rhsOp, printMatrix, "out/rhsOp.txt" );
   
      //
      // define the preconditioner.
      //
      prec = new BlockDiagonalPreconditioner(*blockOffsets);
      prec->SetDiagonalBlock(0, new GSSmoother(B11->SpMat()));
      prec->SetDiagonalBlock(1, new GSSmoother(B22->SpMat()));
   }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      Vector b(x.Size()); b=0.0;
      rhsOp->Mult(x, b);
     
      //update LinearForm C2.
      VsRs->SetTime(GetTime()+deltaT);
      *C2=0;   //need to be seroed since assemble add-on.
      C2->Assemble();

      // Slice the second block (starts at nbrDof)
      int nbrDof = FESpace->GetVSize();
      Vector b_V(b, nbrDof, nbrDof);
      b_V += *C2; // This is the B^a * Vs(t)/Rs term

      //solve for y.
      GMRESSolver solver;
      solver.SetAbsTol(1e-18);
      solver.SetRelTol(1e-18);
      solver.SetMaxIter(400);
      solver.SetPrintLevel(0);
      solver.SetKDim(30);
      solver.SetOperator(*lhsOp);
      solver.SetPreconditioner(*prec);
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
   
   OptionsParser args(argc, argv);
   
   args.AddOption(&L,  "-L",  "--inductance",
                  "Inductance per unit length [H/m].");
   args.AddOption(&C,  "-C",  "--capacitance",
                  "Capacitance per unit length [F/m].");
   args.AddOption(&R,  "-R",  "--resistance",
                  "Resistance per unit length [Ohm/m].");
   args.AddOption(&G,  "-G",  "--conductance",
                  "Conductance per unit length [S/m].");
   args.AddOption(&Rs, "-Rs", "--source-impedance",
                  "Source impedance [Ohm].");
   args.AddOption(&Rl, "-Rl", "--load-impedance",
                  "Load impedance [Ohm].");
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
   args.AddOption(&functionIndex, "-fi", "--functionIndex",
                  "0-gaussian pulse, 1-step 2-sine");

   args.AddOption(&fp1, "-fp1", "--functionparam1",
                  "parameter #1 for source function");
   args.AddOption(&fp2, "-fp2", "--functionparam2",
                  "parameter #2 for source function");
   args.AddOption(&vis, "-vis", "--visualisation",
                  "0-no visualisation, 1-GLVIS 2-gnuplot");

                 
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

   int plotCount = 1; //do not plot at time=0.
   int nbrOfPlot = 5;
   double plotTimeInterval = endTime/nbrOfPlot;
   real_t time = 0.0;
   
   GridFunction *VGF, *IGF;
   VGF = new GridFunction(FESpace);
   IGF = new GridFunction(FESpace);
   // rebuild GFR and GFI from x.
   IGF->MakeRef(FESpace, x, 0);
   VGF->MakeRef(FESpace, x, nbrDof);

   FILE* gnuplotPipe;
   if(vis==2) gnuplotPipe = CreatePlot5x2();

   while(1)
   {
      //Save in a Vector for testing SourceFunction.
      if(sourceFunctionCounter<sourceFunctionVector.Size())
      {
         sourceFunctionVector[sourceFunctionCounter++] = SourceFunction(Zero, time);  
      }

      teleOp->SetTime(time);
      
      Vector xnew(x.Size()); xnew=0.0;
      teleOp->Mult(x, xnew);
      x = xnew;  

      if(time/plotTimeInterval > 1.0*plotCount)
      {
        IGF->MakeRef(FESpace, x, 0);
        VGF->MakeRef(FESpace, x, nbrDof);

        if(vis==1) {
            std::string s = "V after " + std::to_string(time * 1e9) + "ns." ;
            Glvis(mesh, VGF, s, 8, "keys 'caRR'" );
            s = "I after " + std::to_string(time * 1e9) + "ns." ;
            Glvis(mesh, IGF, s, 8, "keys 'caRR'" );
        }
        else if(vis==2)
        {
            std::string VS, IS;
            ExportForGnuplot(*VGF, 11, VS);
            ExportForGnuplot(*IGF, 11, IS);
             
            // Call our internal Gnuplot function
            PlotRow(gnuplotPipe, mesh, VS, IS, time);
        }

         plotCount++;
         if (time>endTime) break;
         if(vis==1)
         {
            cout << "pause" << endl;
            getchar();
         }
      }
      
      time += deltaT;
      }

        if(vis==2) ClosePlot5x2(gnuplotPipe);

         delete VGF;
         delete IGF;
      
         MyPrintFile(&sourceFunctionVector, printMatrix, "out/sourcefunction.txt" );
   
   return 0;
}
