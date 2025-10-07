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

real_t timeScaling = 1e9;

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

        TransposeOperator *BBT;
         ProductOperator *AM1BT;
         ProductOperator *BAM1BT;

         GMRESSolver *solver;
         

   public:
      PB11(BlockOperator &BB_, BlockOperator &BAinv_) :
      Operator(BB_.Height(), BB_.Height()),
      BB(BB_), BAinv(BAinv_)
      {
         assert(BB.Width() == BAinv.Width());
         BBT = new TransposeOperator(BB);
         AM1BT = new ProductOperator(&BAinv, BBT, false, false);
         BAM1BT = new ProductOperator(&BB, AM1BT, false, false);
         solver = new GMRESSolver();
         solver->SetAbsTol(1e-15);
         solver->SetRelTol(1e-5);
         solver->SetMaxIter(5000);
         solver->SetPrintLevel(1);
         solver->SetKDim(50);
         solver->SetOperator(*BAM1BT);
      }

      virtual void Mult(const Vector &x, Vector &y) const override
      {

         solver->Mult(x, y);
         
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

static real_t SourceFunctionGaussianPulse(const Vector x)
{
   /* gaussian pulse of tw wide centered at tc.*/
   real_t t = x(1); //1 for y-axis.
   real_t tw = 20e-9 * timeScaling;
   real_t tc = 20e-9 * timeScaling;
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
   int nbrLengthSeg = 100;
         
   // Constants for the telegrapher’s equation for RG-58, 50 ohm.
   double L = 250e-9 * timeScaling;  // Inductance per unit length
   double C = 100e-12 * timeScaling; // Capacitance per unit length
   double R = 220e-3;  // Resistance per unit length
   double G = 1.0e-7;  // Conductance per unit length

   double Rs = 50.0;
   
   int nbrTimeSeg = 70;
   real_t endTime = 100e-9 * timeScaling;
   real_t deltaT = endTime/nbrTimeSeg;
   real_t Time = 0.0;
//   
// Creates 2D mesh, divided into equal intervals.
//
   Mesh *mesh = new Mesh();
   *mesh = Mesh::MakeCartesian2D(nbrLengthSeg, nbrTimeSeg,
                                  Element::QUADRILATERAL, false, lenght, endTime, false);
   int dim = mesh->Dimension();
   assert(dim==2); //this software works only for dimension 2.
   mesh->PrintInfo();

   int nbrEl = mesh->GetNE();
   assert(nbrEl == nbrLengthSeg * nbrTimeSeg);

   {
      std::ofstream out("out/meshprint.txt");
      if(printMatrix) mesh->Print(out);
   }

   
   //
   // Create the spaces.
   //

   //space for voltage.
   H1_FECollection *VFEC = new H1_FECollection(order, 2);
   FiniteElementSpace *VFESpace = new FiniteElementSpace(mesh, VFEC);
   int VnbrDof = VFESpace->GetTrueVSize(); 
   cout << VnbrDof << " VFESpace degree of freedom\n";  
   assert(VnbrDof == (nbrLengthSeg + 1) * (nbrTimeSeg + 1)); 

   //space for current.
   DG_FECollection *IFEC = new DG_FECollection(order, 2);
   FiniteElementSpace *IFESpace = new FiniteElementSpace(mesh, IFEC);
   int InbrDof = IFESpace->GetTrueVSize(); 
   cout << InbrDof << " IFESpace degree of freedom\n";   
   assert(InbrDof == 4 * nbrLengthSeg * nbrTimeSeg); 
   {
      std::ofstream out("out/ifespace.txt");
      if(printMatrix) IFESpace->Save(out);
   }
   
   //The next three spaces are for application of voltage.
   //space for lagrange multiplier 1 relate to Vs Rs,
   //which cause boundary cnditions on I(0,y).
   
   H1_Trace_FECollection *LM1FEC = new H1_Trace_FECollection(order, 2);
   FiniteElementSpace *LM1FESpace = new FiniteElementSpace(mesh, LM1FEC);
   int LM1nbrDof = LM1FESpace->GetTrueVSize(); 
   cout << LM1nbrDof << " LM1FESpace degree of freedom\n";
   assert(LM1nbrDof == (nbrTimeSeg + 1) * (nbrLengthSeg + 1));

//
//Create the forms.
//
   //the forms for the voltage equation
   //BLF_dvdx implements the x dimension V space derivative,

   ConstantCoefficient one(1.0);
   MixedBilinearForm *MBLF_dvdx = new MixedBilinearForm(VFESpace /*trial*/, VFESpace /*test*/);
   MBLF_dvdx->AddDomainIntegrator(new DerivativeIntegrator(one, 0)); //x direction.
   MBLF_dvdx->Assemble();
   MBLF_dvdx->Finalize();
   cout << MBLF_dvdx->Height() << " MBLF_dvdx Height()." << endl;
   cout << MBLF_dvdx->Width() << " MBLF_dvdx Width()." << endl;
   assert(MBLF_dvdx->Height() == VnbrDof && MBLF_dvdx->Width() == VnbrDof);

   //MBLF_IV implements the y dimension I derivative which is time and
   //the I.
   ConstantCoefficient CC_R(R);
   Vector vLy(2); vLy = 0.0; vLy(1) = L;
   VectorConstantCoefficient CC_Ly(vLy);
   MixedBilinearForm *MBLF_IV = new MixedBilinearForm(IFESpace /*test*/, VFESpace /*trial*/);
   MBLF_IV->AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(CC_Ly));
   MBLF_IV->AddDomainIntegrator(new MixedScalarMassIntegrator(CC_R));
   MBLF_IV->Assemble();
   MBLF_IV->Finalize();
   cout << MBLF_IV->Height() << " MBLF_IV Height()." << endl;
   cout << MBLF_IV->Width() << " MBLF_IV Width()." << endl;
   assert(MBLF_IV->Height() == VnbrDof);
   assert(MBLF_IV->Width() == InbrDof);
   
   
    //the forms for the current equation
   //MBLF_didx implements the x dimension I space derivative,
   MixedBilinearForm *MBLF_didx = new MixedBilinearForm(IFESpace /*trial*/, IFESpace /*test*/);
   MBLF_didx->AddDomainIntegrator(new DerivativeIntegrator(one, 0)); //x direction.
   MBLF_didx->Assemble();
   MBLF_didx->Finalize();
   cout << MBLF_didx->Height() << " MBLF_didx Height()." << endl;
   cout << MBLF_didx->Width() << " MBLF_didx Width()." << endl;
   assert(MBLF_didx->Height() == InbrDof && MBLF_didx->Width() == InbrDof );

   //MBLF_VI implements the y dimension V derivative which is time and
   //the V.
   ConstantCoefficient CC_G(G);
   Vector vCy(2); vCy = 0.0; vCy(1) = -C;
   VectorConstantCoefficient CC_Cy(vCy);
   MixedBilinearForm *MBLF_VI = new MixedBilinearForm(VFESpace, IFESpace);
   MBLF_VI->AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(CC_Cy));
   MBLF_VI->AddDomainIntegrator(new MixedScalarMassIntegrator(CC_G));
   MBLF_VI->Assemble();
   MBLF_VI->Finalize();
   cout << MBLF_VI->Height() << " MBLF_VI Height()." << endl;
   cout << MBLF_VI->Width() << " MBLF_VI Width()." << endl;
   assert(MBLF_VI->Height() == InbrDof);
   assert(MBLF_VI->Width() == VnbrDof);

   //Mixed Bilinear form VL1.
   MixedBilinearForm *MBLF_VL1 = new MixedBilinearForm(VFESpace, LM1FESpace);
   
   ConstantCoefficient oneOverRs(1.0/Rs);

   Array<int> VsrsBdrMarker;
   assert(mesh->bdr_attributes.Size());
   VsrsBdrMarker.SetSize(mesh->bdr_attributes.Max());
   VsrsBdrMarker = 0;
   VsrsBdrMarker[4 - 1] = 1; // along axis x = 0.

   MBLF_VL1->AddBdrFaceIntegrator(new TraceIntegrator(oneOverRs, 1.0), VsrsBdrMarker);
   MBLF_VL1->Assemble();
   MBLF_VL1->Finalize();
   cout << MBLF_VL1->Height() << " MBLF_VL1 Height()." << endl;
   cout << MBLF_VL1->Width() << " MBLF_VL1 Width()." << endl;
   assert(MBLF_VL1->Height() == LM1nbrDof);
   assert(MBLF_VL1->Width() == VnbrDof);

   #ifdef COMMENT

   //Mixed Bilinear form IL1.
   MixedBilinearForm *MBLF_IL1 = new MixedBilinearForm(IvsrsFESpace, LM1FESpace);
   MBLF_IL1->AddDomainIntegrator(new MixedScalarMassIntegrator(one));
   MBLF_IL1->Assemble();
   MBLF_IL1->Finalize();
   cout << MBLF_IL1->Height() << " MBLF_IL1 Height()." << endl;
   cout << MBLF_IL1->Width() << " MBLF_IL1 Width()." << endl;
   assert(MBLF_IL1->Height() == LM1nbrDof);
   assert(MBLF_IL1->Width() == IvsrsnbrDof);
   //create the submesh tp parent operators for the Lagrange multiplier.
   SubmeshOperator *MBLF_IL1_map = new SubmeshOperator(MBLF_IL1->SpMat(), *IFESpace, *IvsrsFESpace, *LM1FESpace);
   cout << MBLF_IL1_map->Height() << " MBLF_IL1_map Height()." << endl;
   cout << MBLF_IL1_map->Width() << " MBLF_IL1_map Width()." << endl;
   assert(MBLF_IL1_map->Height() == LM1nbrDof);
   assert(MBLF_IL1_map->Width() == InbrDof);

   {
      std::ofstream out("out/MBLF_VL1.txt");
      if(printMatrix) MBLF_VL1->SpMat().PrintMatlab(out); // instead of Print()
   }

   {
      std::ofstream out("out/MBLF_IL1.txt");
      if(printMatrix) MBLF_IL1->SpMat().PrintMatlab(out); // instead of Print()
   }
   
   //Mixed Bilinear form VL2.
   MixedBilinearForm *MBLF_VL2 = new MixedBilinearForm(Vt0FESpace, LM2FESpace);
   MBLF_VL2->AddDomainIntegrator(new MixedScalarMassIntegrator(one));
   MBLF_VL2->Assemble();
   MBLF_VL2->Finalize();
   cout << MBLF_VL2->Height() << " MBLF_VL2 Height()." << endl;
   cout << MBLF_VL2->Width() << " MBLF_VL2 Width()." << endl;
   //create the submesh tp parent operators for the Lagrange multiplier.
   SubmeshOperator *MBLF_VL2_map = new SubmeshOperator(MBLF_VL2->SpMat(), *VFESpace, *Vt0FESpace, *LM2FESpace);

   //Mixed Bilinear form IL3.
   MixedBilinearForm *MBLF_IL3 = new MixedBilinearForm(It0FESpace, LM3FESpace);
   MBLF_IL3->AddDomainIntegrator(new MixedScalarMassIntegrator(one));
   MBLF_IL3->Assemble();
   MBLF_IL3->Finalize();
   cout << MBLF_IL3->Height() << " MBLF_IL3 Height()." << endl;
   cout << MBLF_IL3->Width() << " MBLF_IL3 Width()." << endl;
   //create the submesh tp parent operators for the Lagrange multiplier.
   SubmeshOperator *MBLF_IL3_map = new SubmeshOperator(MBLF_IL3->SpMat(), *IFESpace, *It0FESpace, *LM3FESpace);
 
   {
      std::ofstream out("out/MBLF_VL2.txt");
      if(printMatrix) MBLF_VL2->SpMat().PrintMatlab(out); // instead of Print()
   }

   {
      std::ofstream out("out/MBLF_IL3.txt");
      if(printMatrix) MBLF_IL3->SpMat().PrintMatlab(out); // instead of Print()
   }

   LinearForm *LFVS = new LinearForm(LM1FESpace);
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
      Array<int> *BArowOffset = new Array<int>(3);
      (*BArowOffset)[0]=0;
      (*BArowOffset)[1]=MBLF_dvdx->Height(); 
      (*BArowOffset)[2]=MBLF_VI->Height();
      BArowOffset->PartialSum();
      {
         std::ofstream out("out/BArowOffset.txt");
         BArowOffset->Print(out, 10);
      }

      Array<int> *BAcolOffset = new Array<int>(3);
      (*BAcolOffset)[0]=0;
      (*BAcolOffset)[1]=MBLF_dvdx->Width(); 
      (*BAcolOffset)[2]=MBLF_IV->Width();
      BAcolOffset->PartialSum();
      {
         std::ofstream out("out/BAcolOffset.txt");
         BAcolOffset->Print(out, 10);
      }

      BlockOperator *BA = new BlockOperator(*BArowOffset, *BAcolOffset);

      BA->SetBlock(0, 0, MBLF_dvdx);
      BA->SetBlock(0, 1, MBLF_IV);

      BA->SetBlock(1, 0, MBLF_VI);
      BA->SetBlock(1, 1, MBLF_didx);

      
      cout << BA->Height() << " BA->Height()" << endl;
      cout << BA->Width() << " BA->Width()" << endl;

      {
      std::ofstream out("out/BA.txt");
      if(printMatrix) BA->PrintMatlab(out); // instead of Print()
      }

      // 6. Define the constraints inner BlockStructure BB.
      Array<int> *BBrowOffset = new Array<int>(4);
      (*BBrowOffset)[0]=0;
      (*BBrowOffset)[1]=MBLF_VL1_map->Height(); 
      (*BBrowOffset)[2]=MBLF_VL2_map->Height();
      (*BBrowOffset)[3]=MBLF_IL3_map->Height();
      BBrowOffset->PartialSum();
      
      {
         std::ofstream out("out/BBrowOffset.txt");
         BBrowOffset->Print(out, 10);
      }

      Array<int> *BBcolOffset = new Array<int>(3);
      (*BBcolOffset)[0]=0;
      (*BBcolOffset)[1]=MBLF_VL1_map->Width(); 
      (*BBcolOffset)[2]=MBLF_IL1_map->Width();
      BBcolOffset->PartialSum();
      
      {
         std::ofstream out("out/BBcolOffset.txt");
         BBcolOffset->Print(out, 10);
      }

      BlockOperator *BB = new BlockOperator(*BBrowOffset, *BBcolOffset);

      BB->SetBlock(0, 0, MBLF_VL1_map);
      BB->SetBlock(0, 1, MBLF_IL1_map);
      BB->SetBlock(1, 0, MBLF_VL2_map);
      BB->SetBlock(2, 1, MBLF_IL3_map);
      
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
      BlockVector *bBlockA = new BlockVector(*BArowOffset);
      bBlockA->GetBlock(0) = *bV;
      bBlockA->GetBlock(1) = *bI;
      
      // create the x blockvector A.
      //Vector *xV =new Vector(VnbrDof); *xV = 0.0;
      //Vector *xI =new Vector(InbrDof); *xI = 0.0;
      BlockVector *xBlockA = new BlockVector(*BArowOffset);
      xBlockA->GetBlock(0) = 0.0; //*xV;
      xBlockA->GetBlock(1) = 0.0; //*xI;  
      
      Vector &xV = xBlockA->GetBlock(0);
      Vector &xI = xBlockA->GetBlock(1);





      // create the b block vector B.
     // Vector *bL2 = new Vector(MBLF_VL2->Height()); *bL2=0.0;
     // Vector *bL3 = new Vector(MBLF_IL3->Height()); *bL3=0.0;
      BlockVector *bBlockB = new BlockVector(*BBrowOffset);
      bBlockB->GetBlock(0) = *LFVS;
      bBlockB->GetBlock(1) = 0.0; //*bL2;
      bBlockB->GetBlock(2) = 0.0; //*bL3;


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
      if(printMatrix) bBlock->Print(out, 1);
      }


      // Create outer block vector x.
      BlockVector *xBlock = new BlockVector(*OuterBlockOffsets);
      xBlock->GetBlock(0) = *xBlockA;
      xBlock->GetBlock(1) = 0.0;
      
      if(printMatrix)
      {
         std::ofstream out("out/xblock.txt");
         xBlock->Print(out, 1);
      }
//
// Prepare the preconditionner...
//

   PB00 pb00(*BA);
   assert(pb00.Height() == VnbrDof + InbrDof);
   assert(pb00.Width() == VnbrDof + InbrDof);
   
   PB11 pb11(*BB, pb00);
   assert(pb11.Height() == LM1nbrDof + LM2nbrDof + LM3nbrDof);
   assert(pb11.Width() == LM1nbrDof + LM2nbrDof + LM3nbrDof);

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
      solver.SetAbsTol(1e-12);
      solver.SetRelTol(1e-8);
      solver.SetMaxIter(5000);
      solver.SetPrintLevel(1);
      solver.SetKDim(50);
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

   GridFunction *VGF = new GridFunction(VFESpace, xV, 0);
   Glvis(mesh, VGF, "Voltage");

   GridFunction *IGF = new GridFunction(IFESpace, xI, 0);
   Glvis(mesh, IGF, "Current");
#endif //COMMENT
   return 1;
}


