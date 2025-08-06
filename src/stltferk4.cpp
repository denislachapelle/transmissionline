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

As of july 7, 2025 it did not work. The solver do not converge even with square matrix
I mean H1 H1 or L2 L2. I notice the lhsOp converge using octave.
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
        double val = -vs(zeroVec, t) / Rs;
        return -vs(zeroVec, t) / Rs;  // forcing term
    }
};





// TelegrapherOperator is used to timestep, it compute the d(.)/dt.
class TelegrapherOperator : public TimeDependentOperator
{
private:
   
   FiniteElementSpace *VFESpace, *IFESpace;
   MixedBilinearForm *M_VI, *M_IV;
   BilinearForm *S_I, *S_V;
   SparseMatrix *smS_I, *smS_V, *smM_VI, *smM_IV;
   LinearForm *LFVS, *LFVA;
   Array<int> *boundary_dofs;
   Array<int> *bdr_marker;
   VsRsCoefficient *VsRs;

   BlockOperator *lhsOp;
   Array<int> *lhsRowBlockOffset, *lhsColBlockOffset;

   BlockDiagonalPreconditioner *prec;
            
   BlockOperator *rhsOp;
   Array<int> *rhsRowBlockOffset, *rhsColBlockOffset;

    
   // Constants for the telegrapher’s equation for RG-58, 50 ohm.
   double L = 250e-9;  // Inductance per unit length
   double C = 100e-12; // Capacitance per unit length
   double R = 220e-3;  // Resistance per unit length
   double G = 1.0e-2;  // Conductance per unit length

   // source and load impedance.
   double Rs = 50.0;   // source impedance.
   double Rl = 50.0; //load impedance.

public:
   TelegrapherOperator(FiniteElementSpace *VFESpace_, FiniteElementSpace *IFESpace_)
      : TimeDependentOperator(VFESpace_->GetVSize()+IFESpace_->GetVSize(), 0.0, EXPLICIT), VFESpace(VFESpace_), IFESpace(IFESpace_)
   {

      cout << sqrt(L*C) << " sqrt(L*C)\n";      
      //
      // Define the mass and stiffnes matrix
      //
      ConstantCoefficient one(1.0);

      M_IV = new MixedBilinearForm(IFESpace, VFESpace);
      M_IV->AddDomainIntegrator(new MixedScalarMassIntegrator(one));
      M_IV->Assemble();
      M_IV->Finalize();
      smM_IV = &(M_IV->SpMat());

      M_VI = new MixedBilinearForm(VFESpace, IFESpace);
      M_VI->AddDomainIntegrator(new MixedScalarMassIntegrator(one));
      M_VI->Assemble();
      M_VI->Finalize();
      smM_VI = &(M_VI->SpMat());

      S_I = new BilinearForm(IFESpace);
      S_I->AddDomainIntegrator(new DerivativeIntegrator(one, 0));
      S_I->Assemble();
      S_I->Finalize();
      smS_I = &(S_I->SpMat());

      S_V = new BilinearForm(VFESpace);
      S_V->AddDomainIntegrator(new DerivativeIntegrator(one, 0));
      S_V->Assemble();
      S_V->Finalize();
      smS_V = &(S_V->SpMat());

      cout << M_IV->Height() << " M_IV->Height()\n";
      cout << M_IV->Width() << " M_IV->Width()\n";
      
      cout << M_VI->Height() << " M_VI->Height()\n";
      cout << M_VI->Width() << " M_VI->Width()\n";
      
      cout << S_V->Height() << " S_V->Height()\n";
      cout << S_V->Width() << " S_V->Width()\n";

      cout << S_I->Height() << " S_I->Height()\n";
      cout << S_I->Width() << " S_I->Width()\n";

      bdr_marker = new Array<int>(IFESpace->GetMesh()->bdr_attributes.Size());
      assert(bdr_marker->Size() == 2);
      *bdr_marker = 0;
      (*bdr_marker)[0] = 1; 

      if(noSource == false)
      {
         ConstantCoefficient oneOverRs(1.0/Rs);
         LFVA = new LinearForm(IFESpace);
         LFVA->AddBdrFaceIntegrator(new BoundaryLFIntegrator(oneOverRs), *bdr_marker);
         LFVA->Assemble();

         VsRs = new VsRsCoefficient(Rs, SourceFunction);
         VsRs->SetTime(0.0);
         LFVS = new LinearForm(IFESpace);
         LFVS->AddBdrFaceIntegrator(new BoundaryLFIntegrator(*VsRs), *bdr_marker);
         LFVS->Assemble();
      
         {
            std::ofstream out("out/LFVS.txt");
            LFVS->Print(out, 10);
         }

         {
            std::ofstream out("out/LFVA.txt");
            LFVA->Print(out, 10);
         }
      }
      
      // 6. Define the BlockStructure of lhsMatrix
      lhsRowBlockOffset = new Array<int>(3);
      (*lhsRowBlockOffset)[0]=0;
      (*lhsRowBlockOffset)[1]=smM_VI->Height(); 
      (*lhsRowBlockOffset)[2]=smM_IV->Height(); 
      lhsRowBlockOffset->PartialSum();
      {
         std::ofstream out("out/lhsRowBlockOffset.txt");
         lhsRowBlockOffset->Print(out, 10);
      }

      lhsColBlockOffset = new Array<int>(3);
      (*lhsColBlockOffset)[0]=0;
      (*lhsColBlockOffset)[1]=smM_VI->Width(); 
      (*lhsColBlockOffset)[2]=smM_IV->Width(); 
      lhsColBlockOffset->PartialSum();
      {
         std::ofstream out("out/lhsColBlockOffset.txt");
         lhsColBlockOffset->Print(out, 10);
      }
   
      lhsOp = new BlockOperator(*lhsRowBlockOffset, *lhsColBlockOffset);
      
   // Build the operator, insert each block.
   // row 0 ...
         
      lhsOp->SetBlock(0, 0, smM_VI, C);
      lhsOp->SetBlock(1, 1, smM_IV, L);

      {
      std::ofstream out("out/lhsOp.txt");
      if(printMatrix) lhsOp->PrintMatlab(out);
      }

      // 6. Define the BlockStructure of rhsMatrix
      rhsRowBlockOffset = new Array<int>(3);
      (*rhsRowBlockOffset)[0]=0;
      (*rhsRowBlockOffset)[1]=smM_VI->Height(); 
      (*rhsRowBlockOffset)[2]=smS_V->Height(); 
      rhsRowBlockOffset->PartialSum();
      {
         std::ofstream out("out/rhsRowBlockOffset.txt");
         rhsRowBlockOffset->Print(out, 10);
      }

      rhsColBlockOffset = new Array<int>(3);
      (*rhsColBlockOffset)[0]=0;
      (*rhsColBlockOffset)[1]=smM_VI->Width(); 
      (*rhsColBlockOffset)[2]=smS_I->Width(); 
      rhsColBlockOffset->PartialSum();
      {
         std::ofstream out("out/rhsColBlockOffset.txt");
         rhsColBlockOffset->Print(out, 10);
      }

      rhsOp = new BlockOperator(*rhsRowBlockOffset, *rhsColBlockOffset);
      
   // Build the operator, insert each block.
   // rhsOp->SetBlock(0, 0, DofByOne);
      rhsOp->SetBlock(0, 0, smM_VI, -G);
      rhsOp->SetBlock(0, 1, smS_I, 1.0);
      rhsOp->SetBlock(1, 0, smS_V, 1.0);
      rhsOp->SetBlock(1, 1, smM_IV, -R);

      {
      std::ofstream out("out/rhsOp1.txt");
      if(printMatrix) rhsOp->PrintMatlab(out);
      }

      if(noSource == false)
      {
         // add linear form LFVA to row 0 of block 00.
         Operator &A00_copy = rhsOp->GetBlock(0, 0);
         SparseMatrix *SP = dynamic_cast<SparseMatrix*>(&A00_copy);

         for(int i=0; i<LFVA->Size(); i++)
         {
            if ((*LFVA)[i] != 0.0)
            {
               SP->Add(0, i, -(*LFVA)[i]);
            }
         }

         {
         std::ofstream out("out/rhsOp2.txt");
         if(printMatrix) rhsOp->PrintMatlab(out);
         }  
      }

prec = new BlockDiagonalPreconditioner(*lhsRowBlockOffset);

// Example: setting Jacobi or AMG on each block
prec->SetDiagonalBlock(0, new GSSmoother(*smM_VI));
prec->SetDiagonalBlock(1, new GSSmoother(*smM_IV));

      /*
      lhsOpFlat = new SparseMatrix;
      FlattenBlockOperator(*lhsOp, *lhsOpFlat);

      prec = new GSSmoother(*lhsOpFlat);
      */
   }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      Vector b(x.Size());
      rhsOp->Mult(x, b);

      {
         std::ofstream out("out/xrhs.txt");
         if(printMatrix) x.Print(out, 1);
      }  

      {
         std::ofstream out("out/brhs.txt");
         if(printMatrix) b.Print(out, 1);
      }  




      if(noSource == false)
      {
         //update LinearForm LFVS.
         VsRs->SetTime(GetTime());
         *LFVS=0;
         LFVS->Assemble();

         //add LFVS to b.
         Vector b0(b.GetData(), LFVS->Size());
         b0+=*LFVS;
      }

     //DL250707: it look like GMRESSolver cannot solve the operator lhsOp
     //even if octave can. I am stuck.

      /*
      //solve for y.
      GMRESSolver solver;
      solver.SetAbsTol(1e-12);
      solver.SetRelTol(1e-8);
      solver.SetMaxIter(1000);
      solver.SetPrintLevel(1);
      solver.SetKDim(100);
      solver.SetOperator(*lhsOp);
      solver.SetPreconditioner(*prec);
      solver.Mult(b, y);
*/

/*
      //solve for y0.

      GMRESSolver solver;
      solver.SetAbsTol(1e-25);
      solver.SetRelTol(1e-10);
      solver.SetMaxIter(1000);
      solver.SetPrintLevel(1);
      solver.SetKDim(50);
      solver.SetOperator(lhsOp->GetBlock(0, 0));
      //gmres.SetPreconditioner(*prec);
      Vector b0(b.GetData(), IFESpace->GetVSize());
      Vector y0(IFESpace->GetVSize());
      solver.Mult(b0, y0);

      //solve for y1.
      solver.SetAbsTol(1e-25);
      solver.SetRelTol(1e-10);
      solver.SetMaxIter(1000);
      solver.SetPrintLevel(1);
      solver.SetKDim(50);
      solver.SetOperator(lhsOp->GetBlock(1, 1));
      //gmres.SetPreconditioner(*prec);
      Vector b1(b.GetData()+IFESpace->GetVSize(), VFESpace->GetVSize());
      Vector y1(VFESpace->GetVSize());
      solver.Mult(b1, y1);

      Vector out0_view(y.GetData(), IFESpace->GetVSize());
      Vector out1_view(y.GetData() + IFESpace->GetVSize(), VFESpace->GetVSize());
      out0_view = y0;
      out1_view = y1;
   */
   }
};
 

class TransmissionLineTransient
{
   private:
      int order = 1;
      double lenght = 10;
      int nbrSeg = 100;

      real_t deltaT = 0.1e-9;
      real_t endTime = 100e-9;
      real_t Time = 0.0;

      Mesh *mesh;
      int dim; 
      int nbrel;  //number of element.

      FiniteElementCollection *VFEC, *IFEC;
      FiniteElementSpace *VFESpace, *IFESpace;
      int VnbrDof, InbrDof;   //number of degree of freedom.

      TelegrapherOperator *teleOp;

      Vector *I, *V, *b, *y;
      Array<int> *xBlockOffsets; 
      BlockVector *xBlock;  //unknown block vector made of V and I.

      
      GridFunction *VGF, *IGF; 
   
   public:
      TransmissionLineTransient();
      //delete all files in out dir.
      int CleanOutDir();
      //parse the options.
      int Parser(int argc, char *argv[]);
      int LoadMeshFile();
      int CreateFESpace();
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

int TransmissionLineTransient::LoadMeshFile()
{

// Creates 1D mesh, divided into n equal intervals.
   mesh = new Mesh();
   *mesh = mfem::Mesh::MakeCartesian1D(nbrSeg, lenght); 
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

   {
      std::ofstream out("out/meshprint.txt");
      if(printMatrix) mesh->Print(out);
   }
   
   return 1;
}

int TransmissionLineTransient::CreateFESpace()
{
   VFEC = new H1_FECollection(order, dim);
   VFESpace = new FiniteElementSpace(mesh, VFEC);
   VnbrDof = VFESpace->GetTrueVSize(); 
   cout << VnbrDof << " V space degree of freedom\n";   

   IFEC = new H1_FECollection(order, dim);
   IFESpace = new FiniteElementSpace(mesh, IFEC);
   InbrDof = IFESpace->GetTrueVSize(); 
   cout << InbrDof << " I  space degree of freedom\n";   
   return 1;
}

int TransmissionLineTransient::TimeSteps()
{
   teleOp = new TelegrapherOperator(VFESpace, IFESpace);
   RK4Solver solver;
   solver.Init(*teleOp);

   //prepare the xblock [V I]^T.
   I = new Vector(InbrDof); *I=0.0;
   V = new Vector(VnbrDof); *V=1.0;
   
   xBlockOffsets = new Array<int>(3); 
   (*xBlockOffsets)[0]=0;
   (*xBlockOffsets)[1]=VnbrDof;
   (*xBlockOffsets)[2]=InbrDof;
   xBlockOffsets->PartialSum();
   xBlock = new BlockVector(*xBlockOffsets);
   xBlock->GetBlock(0) = *V;
   xBlock->GetBlock(1) = *I;
   
   Vector sourceFunctionVector(endTime/deltaT + 1);
   sourceFunctionVector=0.0;
   int sourceFunctionCounter = 0;
   Vector Zero(3);
   Zero=0.0;

   int plotCount = 0;
   int nbrOfPlot = 3;
   double plotTimeInterval = endTime/nbrOfPlot;

   while(1)
   {
      cout << Time << endl;

      
      //Save in a Vector for testing SourceFunction.
      sourceFunctionVector[sourceFunctionCounter++] = SourceFunction(Zero, Time);  

      //this section of code was quite tested.

      double deltaT_ = deltaT;
      solver.Init(*teleOp);
      solver.Step(*xBlock, Time, deltaT_);

      if(Time/plotTimeInterval >= 1.0*plotCount)
      {
         VGF = new GridFunction(VFESpace);
         IGF = new GridFunction(IFESpace);
  
         // rebuild GFR and GFI from x.
         VGF->MakeRef(VFESpace, *xBlock, 0);
         IGF->MakeRef(IFESpace, *xBlock, VnbrDof);

         std::string s = "V" + std::to_string(plotCount);
         Glvis(mesh, VGF, s, 8, "c" );
         s = "I" + std::to_string(plotCount);
         Glvis(mesh, IGF, s, 8, "c" );

         delete VGF;
         delete IGF;

         plotCount++;


      }
      
      
      if (Time>endTime) break;
    
      
    
   } 

   {
      std::ofstream out("out/sourcefunction.txt");
      if(printMatrix) sourceFunctionVector.Print(out, 1);
   }
   
   return 1;
}

int main(int argc, char *argv[])
{

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, occ::, RAJA and OpenMP based on command line options.
   Device device("cpu");
   device.Print();
   MemoryType mt = device.GetMemoryType();



   TransmissionLineTransient TLT;

   TLT.Parser(argc, argv);
   TLT.CleanOutDir();
   TLT.LoadMeshFile();
   TLT.CreateFESpace();
   TLT.TimeSteps();

   return 0;
}
