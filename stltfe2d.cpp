//                                stltferk4
//                                
// Compile with: make stltfe2d, need MFEM version 4.8 and GLVIS-4.3.
//
// Sample runs:  ./stltfe2d
//
/*
Description:  stltfe (single transmission line transient finite element
2D) simulate a single transmission line with various source
signals, source impedances and load impedances. It treat time as a space dimension
to make use of MFEM capability.
 
Written by Denis Lachapelle, 2025
*/

#include <mfem.hpp>
#include "mytools.hpp"
#include <iostream>
#include <math.h>
#include <filesystem>

using namespace std;
using namespace mfem;

// PB00 preconditionner bloct 00.
class PB00 : public BlockOperator
{
   private:
      // BA Block A of the block operator of the PDE.
      BlockOperator &BA;
      // S00 and S11 are the inverse approximation of the two diagonal block of BA.
      DSmoother *S00, *S11;

   public:
      PB00(BlockOperator &BA_) :
      BlockOperator(BA_.RowOffsets(), BA_.ColOffsets()),
      BA(BA_)
   {
      
      assert(BA_.RowOffsets() == BA_.ColOffsets());
      S00 = new DSmoother(dynamic_cast<BilinearForm&>(BA.GetBlock(0,0)).SpMat());
      S11 = new DSmoother(dynamic_cast<BilinearForm&>(BA.GetBlock(1,1)).SpMat());
      SetBlock(0, 0, S00);
      SetBlock(1, 1, S11);
   }

};

// PB11 preconditioner block 11.
class PB11 : public Operator
{
   private:
      // BB Block operator of the lagrange multiplier.
      // BAinv estimate of the inverse of BA computed with class PB00.
      BlockOperator &BB, &BAinv;
      mutable Vector temp1, temp2;

   public:
      PB11(BlockOperator &BB_, BlockOperator &BAinv_) :
      Operator(BB_.Height(), BB_.Width()),
      BB(BB_), BAinv(BAinv_)
      {

      }

      virtual void Mult(const Vector &x, Vector &y) const override
      {
         BB.MultTranspose(x, temp1);
         BAinv.Mult(temp1, temp2);
         BB.Mult(temp2, y);
      }
};

class Prec : public Solver
{
private:
   PB00 &pb00;
   PB11 &pb11;
   BlockOperator *OuterBlock;
   
public:
   Prec(PB00 &pb00_, PB11 &pb11_)
      : Solver(pb00_.Height()+pb11_.Height(), pb00_.Width()+pb11_.Width()),
        pb00(pb00_), pb11(pb11_)
   {
      Array<int> rowOffsets(3);
      rowOffsets[0]=0;
      rowOffsets[1]=pb00.Height();
      rowOffsets[2]=pb11.Height();
      rowOffsets.PartialSum();

      Array<int> colOffsets(3);
      colOffsets[0]=0;
      colOffsets[1]=pb00.Width();
      colOffsets[2]=pb11.Width();
      colOffsets.PartialSum();

      OuterBlock = new BlockOperator(rowOffsets, colOffsets);
      OuterBlock->SetBlock(0, 0, &pb00);
      OuterBlock->SetBlock(1, 1, &pb11);

   }

   void SetOperator(const Operator &op) 
   {
      // Typically unused for a fixed preconditioner; keep for interface.
      MFEM_VERIFY(op.Height() == Height() && op.Width() == Width(),
                  "Prec::SetOperator size mismatch.");
   }
  
   virtual void Mult(const Vector &x, Vector &y) const override
   {
      OuterBlock->Mult(x, y);
   }
};

void GetDiagFromOperator(const Operator &op, Vector &diag)
{
    int n = op.Height();
    MFEM_VERIFY(n == op.Width(), "must be square");
    diag.SetSize(n);

    Vector x(n), y(n);
    for (int i = 0; i < n; i++)
    {
        x = 0.0; x(i) = 1.0; // unit vector e_i
        op.Mult(x, y);
        diag(i) = y(i);
    }
}

real_t SourceFunctionGaussianPulse(const Vector x)
{
   /* gaussian pulse of tw wide centered at tc.*/
   real_t t = x(1); //1 for y-axis.
   real_t tw = 20e-9;
   real_t tc = 20e-9;
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

FunctionCoefficient VsRsFunctionCoeff(SourceFunctionGaussianPulse);



int main(int argc, char *argv[])
{

   // default options...
   bool printMatrix = true; //control matrix saving to file.
   bool noSource = true; // no code for the source enabled.
   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&printMatrix, "-prm", "--printmatrix", "-dnprm", "--donotprintmatrix",
                  "Print of not the matrix.");
                 
   args.Parse();

   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   
//Erase the content of the /out directory.
   system("rm -f out/*");


   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, occ::, RAJA and OpenMP based on command line options.
   Device device("cpu");
   device.Print();
   MemoryType mt = device.GetMemoryType();


   double lenght = 10;
   int nbrSeg = 10;
         
   // Constants for the telegrapher’s equation for RG-58, 50 ohm.
   double L = 250e-9;  // Inductance per unit length
   double C = 100e-12; // Capacitance per unit length
   double R = 220e-3;  // Resistance per unit length
   double G = 1.0e-7;  // Conductance per unit length

   double Rs = 50.0;
   
   real_t deltaT = 1e-9;
   real_t endTime = 100e-9;
   real_t Time = 0.0;
//   
// Creates 2D mesh, divided into equal intervals.
//
   Mesh *mesh = new Mesh();
   *mesh = Mesh::MakeCartesian2D(nbrSeg, static_cast<int>(endTime / deltaT),
                                  Element::QUADRILATERAL, false, lenght, endTime, false);
   int dim = mesh->Dimension();
   assert(dim==2); //this software works only for dimension 2.
   mesh->PrintInfo();

   int nbrEl = mesh->GetNE();

   {
      std::ofstream out("out/meshprint.txt");
      if(printMatrix) mesh->Print(out);
   }

   //
   //create the submesh for the vsrs boundary.
   //
   Array<int> *vsrs_bdr_attrs = new Array<int>;
   vsrs_bdr_attrs->Append(4); // Attribute ID=4 for the left boundary.
   SubMesh *vsrsmesh = new SubMesh(SubMesh::CreateFromBoundary(*mesh, *vsrs_bdr_attrs));
   //set BdrAttribute to 0 meaning they are not boundary, necessary for LFVS.
   for (int i = 0; i < vsrsmesh->GetNBE(); i++)
   {
        vsrsmesh->SetBdrAttribute(i, 0);
   }
   vsrsmesh->FinalizeMesh();
   vsrsmesh->PrintInfo();

   {
      std::ofstream out("out/vsrsmeshprint.txt");
      if(printMatrix) vsrsmesh->Print(out);
   }
  
   //
   //create the submesh for the t initial boundary.
   //
   Array<int> *t0_bdr_attrs = new Array<int>;
   t0_bdr_attrs->Append(1); // Attribute ID=1 for the bottom boundary.
   SubMesh *t0mesh = new SubMesh(SubMesh::CreateFromBoundary(*mesh, *t0_bdr_attrs));
   //set BdrAttribute to 0 meaning they are not boundary.
   for (int i = 0; i < t0mesh->GetNBE(); i++)
   {
        t0mesh->SetBdrAttribute(i, 0);
   }
   t0mesh->FinalizeMesh();
   t0mesh->PrintInfo();

   {
      std::ofstream out("out/t0meshprint.txt");
      if(printMatrix) t0mesh->Print(out);
   }
  
   //
   // Create the spaces.
   //

   //space for voltage.
   H1_FECollection *VFEC = new H1_FECollection(order, 2);
   FiniteElementSpace *VFESpace = new FiniteElementSpace(mesh, VFEC);
   int VnbrDof = VFESpace->GetTrueVSize(); 
   cout << VnbrDof << " VFESpace degree of freedom\n";   

   //space for current.
   L2_FECollection *IFEC = new L2_FECollection(order, 2);
   FiniteElementSpace *IFESpace = new FiniteElementSpace(mesh, IFEC);
   int InbrDof = IFESpace->GetTrueVSize(); 
   cout << InbrDof << " IFESpace degree of freedom\n";   
   {
      std::ofstream out("out/ifespace.txt");
      if(printMatrix) IFESpace->Save(out);
   }
   
   //space for lagrange multiplier 1 relate to Vs Rs,
   //which cause boundary cnditions on I(0,y).
   
   L2_FECollection *VSRSFEC = new L2_FECollection(order, 1);
   FiniteElementSpace *VSRSFESpace = new FiniteElementSpace(vsrsmesh, VSRSFEC);
   int VSRSnbrDof = VSRSFESpace->GetTrueVSize(); 
   cout << VSRSnbrDof << " ISFESpace degree of freedom\n";   

   //space for lagrange multiplier 2 relate to t initial.
   //which cause boundary cnditions of zero on V(x, 0) and I(x, 0).
   
   L2_FECollection *T0FEC = new L2_FECollection(order, 1);
   FiniteElementSpace *T0FESpace = new FiniteElementSpace(t0mesh, T0FEC);
   int T0nbrDof = T0FESpace->GetTrueVSize(); 
   cout << T0nbrDof << " T0FESpace degree of freedom\n";   

//
//Create the forms.
//
   //the forms for the voltage equation
   //BLF_dvdx implements the x dimension V space derivative,

   ConstantCoefficient one(1.0), Small(0.000000001);
   BilinearForm *BLF_dvdx = new BilinearForm(VFESpace);
   BLF_dvdx->AddDomainIntegrator(new DerivativeIntegrator(one, 0)); //x direction.
   BLF_dvdx->AddDomainIntegrator(new MassIntegrator(Small));
   BLF_dvdx->Assemble();
   BLF_dvdx->Finalize();
   cout << BLF_dvdx->Size() << " BLF_dvdx Size()." << endl;

   //MBLF_IV implements the y dimension I derivative which is time and
   //the I.
   ConstantCoefficient CC_L(L);
   ConstantCoefficient CC_R(R);
   MixedBilinearForm *MBLF_IV = new MixedBilinearForm(IFESpace, VFESpace);
   MBLF_IV->AddDomainIntegrator(new DerivativeIntegrator(CC_L, 1)); //y direction.
   MBLF_IV->AddDomainIntegrator(new MixedScalarMassIntegrator(CC_R));
   MBLF_IV->Assemble();
   MBLF_IV->Finalize();
   cout << MBLF_IV->Height() << " MBLF_IV Height()." << endl;
   cout << MBLF_IV->Width() << " MBLF_IV Width()." << endl;
   
    //the forms for the current equation
   //BLF_didx implements the x dimension I space derivative,
   BilinearForm *BLF_didx = new BilinearForm(IFESpace);
   BLF_didx->AddDomainIntegrator(new DerivativeIntegrator(one, 0)); //x direction.
   BLF_didx->AddDomainIntegrator(new MassIntegrator(Small));
   BLF_didx->Assemble();
   BLF_didx->Finalize();
   cout << BLF_didx->Size() << " BLF_didx Size()." << endl;

   //MBLF_VI implements the y dimension V derivative which is time and
   //the V.
   ConstantCoefficient CC_C(C);
   ConstantCoefficient CC_G(G);
   MixedBilinearForm *MBLF_VI = new MixedBilinearForm(VFESpace, IFESpace);
   MBLF_VI->AddDomainIntegrator(new DerivativeIntegrator(CC_C, 1)); //y direction.
   MBLF_VI->AddDomainIntegrator(new MixedScalarMassIntegrator(CC_G));
   MBLF_VI->Assemble();
   MBLF_VI->Finalize();
   cout << MBLF_VI->Height() << " MBLF_VI Height()." << endl;
   cout << MBLF_VI->Width() << " MBLF_VI Width()." << endl;
   

   //Mixed Bilinear form VL1.
   MixedBilinearForm *MBLF_VL1 = new MixedBilinearForm(VFESpace, VSRSFESpace);
   ConstantCoefficient oneOverRs(1.0/Rs);
   MBLF_VL1->AddDomainIntegrator(new MixedScalarMassIntegrator(oneOverRs));
   MBLF_VL1->Assemble();
   MBLF_VL1->Finalize();
   cout << MBLF_VL1->Height() << " MBLF_VL1 Height()." << endl;
   cout << MBLF_VL1->Width() << " MBLF_VL1 Width()." << endl;
   

   //Mixed Bilinear form IL1.
   MixedBilinearForm *MBLF_IL1 = new MixedBilinearForm(IFESpace, VSRSFESpace);
   MBLF_IL1->AddDomainIntegrator(new MixedScalarMassIntegrator(one));
   MBLF_IL1->Assemble();
   MBLF_IL1->Finalize();
   cout << MBLF_IL1->Height() << " MBLF_IL1 Height()." << endl;
   cout << MBLF_IL1->Width() << " MBLF_IL1 Width()." << endl;
 
   {
      std::ofstream out("out/MBLF_VL1.txt");
      if(printMatrix) MBLF_VL1->SpMat().PrintMatlab(out); // instead of Print()
   }

   {
      std::ofstream out("out/MBLF_IL1.txt");
      if(printMatrix) MBLF_IL1->SpMat().PrintMatlab(out); // instead of Print()
   }
   
   //Mixed Bilinear form VL2.
   MixedBilinearForm *MBLF_VL2 = new MixedBilinearForm(VFESpace, T0FESpace);
   MBLF_VL2->AddDomainIntegrator(new MixedScalarMassIntegrator(one));
   MBLF_VL2->Assemble();
   MBLF_VL2->Finalize();
   cout << MBLF_VL2->Height() << " MBLF_VL2 Height()." << endl;
   cout << MBLF_VL2->Width() << " MBLF_VL2 Width()." << endl;
   
   //Mixed Bilinear form IL3.
   MixedBilinearForm *MBLF_IL3 = new MixedBilinearForm(IFESpace, T0FESpace);
   MBLF_IL3->AddDomainIntegrator(new MixedScalarMassIntegrator(one));
   MBLF_IL3->Assemble();
   MBLF_IL3->Finalize();
   cout << MBLF_IL3->Height() << " MBLF_IL3 Height()." << endl;
   cout << MBLF_IL3->Width() << " MBLF_IL3 Width()." << endl;
 
   {
      std::ofstream out("out/MBLF_VL2.txt");
      if(printMatrix) MBLF_VL2->SpMat().PrintMatlab(out); // instead of Print()
   }

   {
      std::ofstream out("out/MBLF_IL3.txt");
      if(printMatrix) MBLF_IL3->SpMat().PrintMatlab(out); // instead of Print()
   }

   LinearForm *LFVS = new LinearForm(VSRSFESpace);
   LFVS->AddDomainIntegrator(new DomainLFIntegrator(VsRsFunctionCoeff));
   LFVS->Assemble();

   cout << LFVS->Size() << " LFVS Size()." << endl;
 
   {
      std::ofstream out("out/lfvs.txt");
      if(printMatrix) LFVS->Print(out, 1); // instead of Print()
   }

//
//Define the inner and outer blockopertor.
//

   // 6. Define the PDE inner BlockStructure BA.
      Array<int> *BAOffset = new Array<int>(3);
      (*BAOffset)[0]=0;
      (*BAOffset)[1]=BLF_dvdx->Size(); 
      (*BAOffset)[2]=MBLF_VI->Height();
      BAOffset->PartialSum();
      {
         std::ofstream out("out/BAOffset.txt");
         BAOffset->Print(out, 10);
      }

      BlockOperator *BA = new BlockOperator(*BAOffset);

      BA->SetBlock(0, 0, BLF_dvdx);
      BA->SetBlock(0, 1, MBLF_IV);

      BA->SetBlock(1, 0, MBLF_VI);
      BA->SetBlock(1, 1, BLF_didx);

      
      cout << BA->Height() << " BA->Height()" << endl;
      cout << BA->Width() << " BA->Width()" << endl;

      {
      std::ofstream out("out/BA.txt");
      if(printMatrix) BA->PrintMatlab(out); // instead of Print()
      }

      // 6. Define the constraints inner BlockStructure BB.
      Array<int> *BBrowOffset = new Array<int>(4);
      (*BBrowOffset)[0]=0;
      (*BBrowOffset)[1]=MBLF_VL1->Height(); 
      (*BBrowOffset)[2]=MBLF_VL2->Height();
      (*BBrowOffset)[3]=MBLF_IL3->Height();
      BBrowOffset->PartialSum();
      
      {
         std::ofstream out("out/BBrowOffset.txt");
         BAOffset->Print(out, 10);
      }

      Array<int> *BBcolOffset = new Array<int>(3);
      (*BBcolOffset)[0]=0;
      (*BBcolOffset)[1]=MBLF_VL1->Width(); 
      (*BBcolOffset)[2]=MBLF_IL1->Width();
      BBcolOffset->PartialSum();
      
      {
         std::ofstream out("out/BBcolOffset.txt");
         BAOffset->Print(out, 10);
      }

      BlockOperator *BB = new BlockOperator(*BBrowOffset, *BBcolOffset);

      BB->SetBlock(0, 0, MBLF_VL1);
      BB->SetBlock(0, 1, MBLF_IL1);
      BB->SetBlock(1, 0, MBLF_VL2);
      BB->SetBlock(2, 1, MBLF_IL3);
      
      cout << BB->Height() << " BB->Height()" << endl;
      cout << BB->Width() << " BB->Width()" << endl;

      {
      std::ofstream out("out/BB.txt");
      if(printMatrix) BB->PrintMatlab(out); // instead of Print()
      }

      //
      // create the outer block structure composed of the inner block.
      //
           
      
      Array<int> *OuterBlockOffsets = new Array<int>(3);
      (*OuterBlockOffsets)[0]=0;
      (*OuterBlockOffsets)[1]=BA->Height(); 
      (*OuterBlockOffsets)[2]=BB->Height();
      OuterBlockOffsets->PartialSum();
      
      {
         std::ofstream out("out/OuterBlockOffsets.txt");
         OuterBlockOffsets->Print(out, 10);
      }

      BlockOperator *OuterBlock = new BlockOperator(*OuterBlockOffsets);

      TransposeOperator *BBT = new TransposeOperator(BB);

      OuterBlock->SetBlock(0, 0, BA);
      OuterBlock->SetBlock(0, 1, BBT);
      OuterBlock->SetBlock(1, 0, BB);
      
      cout << OuterBlock->Height() << " OuterBlock->Height()" << endl;
      cout << OuterBlock->Width() << " OuterBlock->Width()" << endl;

      {
      std::ofstream out("out/OuterBlock.txt");
      if(printMatrix) OuterBlock->PrintMatlab(out); // instead of Print()
      }

   //
   //create the x and b vectors.
   //   
      // create the b blockvector A.
      Vector *bV = new Vector(VnbrDof); *bV = 0.0;
      Vector *bI = new Vector(InbrDof); *bI = 0.0;
      BlockVector *bBlockA = new BlockVector(*BAOffset);
      bBlockA->GetBlock(0) = *bV;
      bBlockA->GetBlock(1) = *bI;
      
      // create the x blockvector A.
      Vector *xV =new Vector(VnbrDof); *xV = 0.0;
      Vector *xI =new Vector(InbrDof); *xI = 0.0;
      BlockVector *xBlockA = new BlockVector(*BAOffset);
      xBlockA->GetBlock(0) = *xV;
      xBlockA->GetBlock(1) = *xI;   

      // create the b block vector B.
      Vector *bL2 = new Vector(MBLF_VL2->Height()); *bL2=0.0;
      Vector *bL3 = new Vector(MBLF_IL3->Height()); *bL3=0.0;
      BlockVector *bBlockB = new BlockVector(*BBrowOffset);
      bBlockB->GetBlock(0) = *LFVS;
      bBlockB->GetBlock(1) = *bL2;
      bBlockB->GetBlock(2) = *bL3;

      {
      std::ofstream out("out/bblockb.txt");
      if(printMatrix) bBlockB->Print(out, 1); // instead of Print()
      }

      // Create outer block vector b.
      BlockVector *bBlock = new BlockVector(*OuterBlockOffsets);
      bBlock->GetBlock(0) = *bBlockA;
      bBlock->GetBlock(1) = *bBlockB;
      
      {
      std::ofstream out("out/bblock.txt");
      if(printMatrix) bBlock->Print(out, 1); // instead of Print()
      }


      // Create outer block vector x.
      BlockVector *xBlock = new BlockVector(*OuterBlockOffsets);
      xBlock->GetBlock(0) = *xBlockA;

//
// Prepare the preconditionner...
//

   PB00 pb00(*BA);
   PB11 pb11(*BB, pb00);
   Prec prec(pb00, pb11);

//PrecOperator *prec = new PrecOperator(*OuterBlock, *BA, *BB);


   /*

   
      MinvBt = TransposeOperator(BB);

      for (int i = 0; i < Ad.Size(); i++)
      {
         MinvBt->ScaleRow(i, 1./Ad(i));
      }

      S = Mult(B, *MinvBt);

      invA = new DSmoother(A);


      invS = new GSSmoother(*S);
  

   invA->iterative_mode = false;
   invS->iterative_mode = false;

   prec.SetDiagonalBlock(0, invM);
   prec.SetDiagonalBlock(1, invS);
*/
//
// Solvethe equations system.
//
   if(1)
   {
      GMRESSolver solver;
      solver.SetAbsTol(1e-32);
      solver.SetRelTol(1e-8);
      solver.SetMaxIter(5000);
      solver.SetPrintLevel(1);
      solver.SetKDim(5000);
      solver.SetOperator(*OuterBlock);
      solver.SetPreconditioner(prec);
      solver.Mult(*bBlock, *xBlock);
   }

   if(0)
   {
      MINRESSolver solver;
   solver.SetAbsTol(1e-20);
   solver.SetRelTol(1e-8);
   solver.SetMaxIter(5000);
   solver.SetOperator(*OuterBlock);
   //solver.SetPreconditioner(*prec);
   solver.SetPrintLevel(1);
   solver.Mult(*bBlock, *xBlock);
   }

   GridFunction *VGF = new GridFunction(VFESpace, *xV, 0);
   Glvis(mesh, VGF, "Voltage");

   GridFunction *IGF = new GridFunction(IFESpace, *xI, 0);
   Glvis(mesh, IGF, "Current");

   return 1;
}


