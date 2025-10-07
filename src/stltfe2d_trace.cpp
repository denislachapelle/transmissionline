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
#include <mesh/submesh/submesh.hpp>
#include <mesh/submesh/transfermap.hpp>
#include "mytools.hpp"
#include <iostream>
#include <math.h>
#include <filesystem>

using namespace std;
using namespace mfem;

double timeScaling = 1e9; // timeScaling to make all number in the same order.

// Adapter: make TransferMap look like an Operator (n_sub × n_parent)
class TransferMapOp : public Operator
{
private:
  FiniteElementSpace *src_fes, *dst_fes;
  TransferMap *Tmap;
public:
  TransferMapOp(FiniteElementSpace *src_fes_,
                FiniteElementSpace *dst_fes_)
  : Operator(dst_fes_->GetVSize(), src_fes_->GetVSize()),
    src_fes(src_fes_), dst_fes(dst_fes_)
    {
       Tmap =  new TransferMap(*src_fes, *dst_fes);
    }

  void Mult(const Vector &x, Vector &y) const override
  {
     GridFunction gsrc(src_fes), gdst(dst_fes);
     gsrc = x;
     Tmap->Transfer(gsrc, gdst); // y = gy;
  }
  /*
  void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const override {
    // NOTE: this uses the reverse TransferMap as an adjoint surrogate.
    mfem::GridFunction gx(&dst_fes_), gy(&src_fes_);
    gx = x; reverse_.Transfer(gx, gy); y = gy;
  }
    */
};

// PB00 preconditionner bloct 00.
class PB00 : public BlockOperator
{
   private:
      // BA Block A of the block operator of the PDE.
      BlockOperator &BA;
      // S00 and S11 are the inverse approximation of the two diagonal block of BA.
      DSmoother  *S00, *S11;

   public:
      PB00(BlockOperator &BA_) :
      BlockOperator(BA_.RowOffsets(), BA_.ColOffsets()),
      BA(BA_)
   {
      
      assert(BA_.RowOffsets() == BA_.ColOffsets());
      S00 = new DSmoother (dynamic_cast<BilinearForm&>(BA.GetBlock(0,0)).SpMat());
      S11 = new DSmoother (dynamic_cast<BilinearForm&>(BA.GetBlock(1,1)).SpMat());
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
      Operator(BB_.Height(), BB_.Height()),
      BB(BB_), BAinv(BAinv_)
      {
         temp1.SetSize(BB.Width());
         temp2.SetSize(BB.Width());
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
   real_t tw = 20e-9 * timeScaling;
   real_t tc = 50e-9 * timeScaling;
   if(t<2*tc) return 4.0 * t/(2.0*tc)*(1-t/(2.0*tc)) * exp(-pow(((t-tc)/tw), 2.0));
   else return 0.0;
}

real_t SourceFunctionStep(const Vector x)
{
      //step.
      real_t t = x(1); //1 for y-axis.
      real_t tau = 30e-9;
      return 1.0 - exp(-t/tau);
}

real_t SourceFunctionSine(const Vector x, real_t t)
{
      return sin(2*M_PI*13e6*t);
}

FunctionCoefficient LMFunctionCoeff(SourceFunctionStep);



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
   double L = 250e-9 * timeScaling;  // Inductance per unit length, -9
   double C = 100e-12 * timeScaling; // Capacitance per unit length, -12
   double R = 220e-3;  // Resistance per unit length
   double G = 1.0e-7;  // Conductance per unit length

   double Rs = 50.0;
   
   real_t deltaT = 10e-9 * timeScaling;
   real_t endTime = 100e-9 * timeScaling;
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
   // Create the spaces.
   //

   //space for voltage.
   H1_FECollection *VFEC = new H1_FECollection(order, dim);
   FiniteElementSpace *VFESpace = new FiniteElementSpace(mesh, VFEC);
   int VnbrDof = VFESpace->GetTrueVSize(); 
   cout << VnbrDof << " VFESpace degree of freedom\n";   

   //space for current.
   H1_FECollection *IFEC = new H1_FECollection(order, dim);
   FiniteElementSpace *IFESpace = new FiniteElementSpace(mesh, IFEC);
   int InbrDof = IFESpace->GetTrueVSize(); 
   cout << InbrDof << " IFESpace degree of freedom\n";   
   {
      std::ofstream out("out/ifespace.txt");
      if(printMatrix) IFESpace->Save(out);
   }
   
   //space for lagrange multipliers.
   
   H1_Trace_FECollection *LMFEC = new H1_Trace_FECollection(order, dim);
   FiniteElementSpace *LMFESpace = new FiniteElementSpace(mesh, LMFEC);
   int LMnbrDof = LMFESpace->GetTrueVSize(); 
   cout << LMnbrDof << " LMFESpace degree of freedom\n";   
   cout << LMFESpace->GetNDofs() << " Number of trace DOFs" << endl;
   {
      std::ofstream out("out/LMFESpace.txt");
      if(printMatrix) LMFESpace->Save(out);
   }   
//
//Create the forms.
//
   //the forms for the voltage equation
   //BLF_dvdx implements the x dimension V space derivative,

   ConstantCoefficient one(1.0), Small(0.0000000001);
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
   Array<int> *LM_bdr_attrs = new Array<int>;
   LM_bdr_attrs->Append(4); // Attribute ID=4 for the left boundary.
   BilinearForm *MBLF_VL1 = new BilinearForm(LMFESpace);
   ConstantCoefficient oneOverRs(1.0/Rs);
   MBLF_VL1->AddDomainIntegrator(new BoundaryMassIntegrator(oneOverRs), *LM_bdr_attrs);
   MBLF_VL1->Assemble();
   MBLF_VL1->Finalize();
   cout << MBLF_VL1->Height() << " MBLF_VL1 Height()." << endl;
   cout << MBLF_VL1->Width() << " MBLF_VL1 Width()." << endl;


   //Mixed Bilinear form IL1.

   MixedBilinearForm *MBLF_IL1 = new MixedBilinearForm(LMFESpace, IFESpace);
   MBLF_IL1->AddDomainIntegrator(new BoundaryMassIntegrator(one), *LM_bdr_attrs);
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
   Array<int> *t0_bdr_attrs = new Array<int>;
   t0_bdr_attrs->Append(1); // Attribute ID=1 for the bottom boundary.
   MixedBilinearForm *MBLF_VL2 = new MixedBilinearForm(LMFESpace, VFESpace);
   MBLF_VL2->AddDomainIntegrator(new BoundaryMassIntegrator(one), *t0_bdr_attrs );
   MBLF_VL2->Assemble();
   MBLF_VL2->Finalize();
   cout << MBLF_VL2->Height() << " MBLF_VL2 Height()." << endl;
   cout << MBLF_VL2->Width() << " MBLF_VL2 Width()." << endl;
   
   //Mixed Bilinear form IL3.
   MixedBilinearForm *MBLF_IL3 = new MixedBilinearForm(LMFESpace, IFESpace);
   MBLF_IL3->AddDomainIntegrator(new BoundaryMassIntegrator(one), *t0_bdr_attrs);
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

   LinearForm *LFVS = new LinearForm(LMFESpace);
   LFVS->AddDomainIntegrator(new DomainLFIntegrator(LMFunctionCoeff));
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
         BBrowOffset->Print(out, 10);
      }

      Array<int> *BBcolOffset = new Array<int>(3);
      (*BBcolOffset)[0]=0;
      (*BBcolOffset)[1]=MBLF_VL1->Width(); 
      (*BBcolOffset)[2]=MBLF_IL1->Width();
      BBcolOffset->PartialSum();
      
      {
         std::ofstream out("out/BBcolOffset.txt");
         BBcolOffset->Print(out, 10);
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

      TransposeOperator *BBT = new TransposeOperator(*BB);

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
   //create the x vector.
   //   
      Vector *x = new Vector(VnbrDof + InbrDof + 3*LMnbrDof); *x = 0.0;
      
      Vector *b = new Vector(VnbrDof + InbrDof + 3*LMnbrDof); *b = 0.0;
      b->AddSubVector(*LFVS, VnbrDof + InbrDof);

      
      /*
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
      Vector *bL2 = new Vector(T0nbrDof); *bL2=0.0;
      Vector *bL3 = new Vector(T0nbrDof); *bL3=0.0;
      BlockVector *bBlockB = new BlockVector(*BBrowOffset);
      bBlockB->GetBlock(0) = *LFVS;
      bBlockB->GetBlock(1) = *bL2;
      bBlockB->GetBlock(2) = *bL3;

      {
      std::ofstream out("out/bblockb.txt");
      if(printMatrix) bBlockB->Print(out, 1); // instead of Print()
      }

      // create the x block vector B.const SparseMatrix *LMMap = dynamic_cast<const SparseMatrix*>(LMFESpace->GetProlongationMatrix());
      Vector *xL1 = new Vector(LMnbrDof); *xL1=0.0;
      Vector *xL2 = new Vector(T0nbrDof); *xL2=0.0;
      Vector *xL3 = new Vector(T0nbrDof); *xL3=0.0;
      BlockVector *xBlockB = new BlockVector(*BBrowOffset);
      xBlockB->GetBlock(0) = *xL1;
      xBlockB->GetBlock(1) = *xL2;
      xBlockB->GetBlock(2) = *xL3;

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
      xBlock->GetBlock(1) = *xBlockB;
*/

//
// Prepare the preconditionner...
//

   PB00 pb00(*BA);
   assert(pb00.Height() == VnbrDof + InbrDof);
   assert(pb00.Width() == VnbrDof + InbrDof);
   
   PB11 pb11(*BB, pb00);
   assert(pb11.Height() == 3*LMnbrDof);
   assert(pb11.Width() == 3*LMnbrDof);
   
   Prec prec(pb00, pb11);
   assert(prec.Height() == VnbrDof + InbrDof + 3*LMnbrDof);
   assert(prec.Width() == VnbrDof + InbrDof + 3*LMnbrDof);   

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
      solver.SetAbsTol(1e-15);
      solver.SetRelTol(1e-8);
      solver.SetMaxIter(5000);
      solver.SetPrintLevel(1);
      solver.SetKDim(30);
      solver.SetOperator(*OuterBlock);
      //solver.SetPreconditioner(prec);
      solver.Mult(*b, *x);
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
   solver.Mult(*b, *x);
   }

   GridFunction *VGF = new GridFunction(VFESpace, *x, 0);
   Glvis(mesh, VGF, "Voltage");

   GridFunction *IGF = new GridFunction(IFESpace, *x, VnbrDof);
   Glvis(mesh, IGF, "Current");

   return 0;
}


