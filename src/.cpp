//                                tltransient
//                                
// Compile with: make tltransient, need MFEM version 4.7 and GLVIS-4.3.
//
// Sample runs:  ./tltransient
//
/*
Description:  

*/

#include <mfem.hpp>
//#include <linalg/hypre.hpp>
//#include "mytools.hpp"
#include <iostream>
#include <math.h>
#include <filesystem>
//#include "twodwiresgenerator.hpp"

//Physical group expected from mesh file.
#define input 3
#define output 4
#define stubend 5
#define tl1 1
#define tl2 2


using namespace std;
using namespace mfem;



class TransmissionLineTransient
{

   private:
      // 1. Parse command-line options.
      const char *meshFile = "tlmesh.msh";
      int order = 2;
      bool printMatrix = false;
     
      Mesh *mesh;
      int dim; 
      int nbrel;  //number of element.

      FiniteElementCollection *VFEC, *IFEC;
      FiniteElementSpace *VFESpace, IFESpace;
      int nbrdof;   //number of degree of freedom.

      Array<int> *Vess_tdof_list, *Vess_tdof_list; //essential dof list.

      //matrix pointer and matrix array pointer
      Vector *rhs, *x;

      Operator *A_ptr;
      BlockVector *B, *X;
      
      GSSmoother **gs;
      BlockDiagonalPreconditioner *block_prec;

      GridFunction *VGF, *IGF; 
      

   public:

      //delete all files in out dir.
      int CleanOutDir();
      //parse the options.
      int Parser(int argc, char *argv[]);
      int LoadMeshFile();
      int CreateFESpace();
      int CreateEssentialBoundary();
      int CreateOperatorA1();
      int CreateOperatorA2();
      int CreateOperatorA3();
      int CreateOperatorA4();
      int CreateOperatorA5();
      int CreateOperatorA6();
      int CreateOperatorA7();
      int CreaterhsVector();
      int CreatexVector();
      int CreateBlockOperator();
      int CreatePreconditionner();
      int Solver();
      int PostPrecessing();
      int DisplayResults();
      int Save();
};

int TransmissionLineTransient::Parser(int argc, char *argv[])
{

   OptionsParser args(argc, argv);
   args.AddOption(&meshFile, "-mf", "--meshfile",
                  "file to use as mesh file.");
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

int TransmissionLineTransient::LoadMeshFile()
{

   // 3. Read the mesh from the given mesh file.
//   if(IsCreatedMeshFile()) mesh = new Mesh("mesh.msh", 1, 1);
//   else 
   mesh = new Mesh(meshFile, 1, 1);
   
   dim = mesh->Dimension();
   assert(dim==2); //this software works only for dimension 2.

   cout << mesh->bdr_attributes.Max() << " bdr attr max\n"
        << mesh->Dimension() << " dimensions\n"
        << mesh->GetNV() << " vertices\n"
        << mesh->GetNE() << " elements\n"
        << mesh->GetNBE() << " boundary elements\n"
        << mesh->GetNEdges() << " edges\n";

   nbrel = mesh->GetNE();

   return 1;
}

int TransmissionLineTransient::CreateFESpace()
{
   VFEC = new H1_FECollection(order, dim);
   VFESpace = new FiniteElementSpace(mesh, VFEC);
   nbrdof = VFESpace->GetNDofs(); 
   cout << "VFESpace\n" << VFESpace->GetNDofs() << " degree of freedom\n"
        << VFESpace->GetVDim() << " vectors dimension\n\n";   

   IFEC = new DG_FECollection(order, dim);
   IFESpace = new FiniteElementSpace(mesh, IFEC);
   nbrdof = IFESpace->GetNDofs(); 
   cout << "IFESpace\n" << IFESpace->GetNDofs() << " degree of freedom\n"
        << IFESpace->GetVDim() << " vectors dimension\n\n";   


return 1;
}

int TransmissionLineTransient::CreateEssentialBoundary()
{
   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   // real and imag are the same because they refer to mesh nodes.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   assert(mesh->bdr_attributes.Max()==nbrGroups+2);
   ess_bdr.SetSize(nbrGroups+2);
   ess_bdr = 0;
   ess_bdr[AIRCONTOUR-1]=1;
   Afespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   {
      std::ofstream out("out/ess_tdof_list.txt");
      ess_tdof_list.Print(out, 10);
   }

//duplicate the essential boundary condition for the imaginary part of Az.
//it shall be + nbrdof to align with the second block.
   ess_tdof_list_block = new Array<int>(2*ess_tdof_list.Size());
   for(int i=0, size=ess_tdof_list.Size(); i<size; i++)
   {
      (*ess_tdof_list_block)[i]=ess_tdof_list[i];
      (*ess_tdof_list_block)[i+size]=ess_tdof_list[i]+nbrdof;
   }

   {
      std::ofstream out("out/ess_tdof_list_block.txt");
      ess_tdof_list_block->Print(out, 10);
   }
   return 1;
}

int TransmissionLineTransient::CreateOperatorA1()
{
   ConstantCoefficient One(1.0);
// note DiffusionIntegrator is "- div(grad(.))".
   BilinearForm BLFA1(Afespace);
   BLFA1.AddDomainIntegrator(new DiffusionIntegrator(One));
   BLFA1.Assemble();
   BLFA1.Finalize();
   
   A1 = new SparseMatrix(BLFA1.SpMat());

   std::ofstream out("out/A1.txt");
   if(printMatrix) A1->Print(out, 10);
  
   cout << A1->Height() << " A1 Height()\n " 
        << A1->Width()  << " A1 Width()\n\n ";

   return 1;
}

int TransmissionLineTransient::CreateOperatorA2()
{
   
   PWConstCoefficient K(2+nbrGroups);
   Vector CoeffVector(2+nbrGroups);
   CoeffVector = 0.0;
   for(int gc = 0; gc < nbrGroups; gc++)
   {
      CoeffVector[gc] = -mu_*omega_*sigma_;
   }   
   K.UpdateConstants(CoeffVector);

   BilinearForm BLFA2(Afespace);
   BLFA2.AddDomainIntegrator(new MassIntegrator(K));
   BLFA2.Assemble();
   BLFA2.Finalize();
   
   A2 = new SparseMatrix(BLFA2.SpMat());

   std::ofstream out("out/A2.txt");
   if(printMatrix) A2->Print(out, 10);
  
   cout << A2->Height() << " A2 Height()\n " 
        << A2->Width()  << " A2 Width()\n\n ";

   return 1;
}

int TransmissionLineTransient::CreateOperatorA3()
{  
   // point to an array of sparse matrix.
   A3 = new SparseMatrix*[nbrGroups];
   // used to assemble filename.
   stringstream ss;
   char fn[256];
   
   for(int gc = 0; gc < nbrGroups; gc++)
   {
      Vector CoeffVector(2+nbrGroups);
      CoeffVector = 0.0;
      CoeffVector[gc] = -mu_*sigma_;
      PWConstCoefficient K(CoeffVector);

      BilinearForm BLFA3(Afespace);
      BLFA3.AddDomainIntegrator(new MassIntegrator(K));
      BLFA3.Assemble();
      BLFA3.Finalize();
      
      sprintf(fn,"out/BLFA3_%d.txt", gc);
      std::ofstream out4(fn);
      if(printMatrix) BLFA3.SpMat().Print(out4, 10);
      
      SparseMatrix TempSM(BLFA3.SpMat());
      TempSM.Finalize();
      
      Vector TempVEC(nbrdof);
      TempSM.GetRowSums(TempVEC);

      A3[gc] = new SparseMatrix(nbrdof, 1);
      *(A3[gc]) = 0.0;
      
      for(int i=0; i<nbrdof; i++)
      {
         A3[gc]->Add(i, 0, TempVEC[i]);
      }

      A3[gc]->Finalize();

      sprintf(fn,"out/A3_%d.txt", gc );
      std::ofstream out3(fn);
      if(printMatrix) A3[gc]->Print(out3, 10);
      
      cout << A3[gc]->Height() << " A3 Height()\n " 
         << A3[gc]->Width()  << " A3 Width()\n\n ";
   }
   return 1;
}


int TransmissionLineTransient::CreateOperatorA4()
{
   A4 = new SparseMatrix(*A2);
   *A4 *= -1.0;
   A4->Finalize();
      
   std::ofstream out("out/A4.txt");
   if(printMatrix) A4->Print(out, 10);

   cout << A4->Height() << " A4 Height()\n " 
        << A4->Width()  << " A4 Width()\n\n ";

   return 1;
}

int TransmissionLineTransient::CreateOperatorA5()
{
/*
This section of code compute the operator performing the 
integration.

For each element compute the current which is the integral of J x s.
Note s, the conductivity, is a PWCoefficient
*/
   A5 = new SparseMatrix*[nbrGroups];
   stringstream ss; 
   char fn[256];
   for(int gc = 0; gc < nbrGroups; gc++)
   {
      Vector CoeffVector(2+nbrGroups);
      CoeffVector = 0.0;
      CoeffVector[gc] = omega_*sigma_;
      PWConstCoefficient K(CoeffVector);
         
      //surface integral.
      LinearForm LFA5(Afespace);
      LFA5.AddDomainIntegrator(new DomainLFIntegrator(K));
      LFA5.Assemble();
      
      sprintf(fn,"out/LFA5_%d.txt", gc );
      std::ofstream out1(fn);
      if(printMatrix) LFA5.Print(out1, 10);

      A5[gc] = new SparseMatrix(1, nbrdof);
      for(int k=0; k<nbrdof; k++)
      {
         A5[gc]->Set(0, k, LFA5[k]);
      }

      A5[gc]->Finalize();

      sprintf(fn,"out/A5_%d.txt", gc );
      std::ofstream out2(fn);
      if(printMatrix) A5[gc]->Print(out2, 10);

      cout << A5[gc]->Height() << " A5 Height()\n " 
         << A5[gc]->Width()  << " A5 Width()\n\n ";
   }

   return 1;
}


int TransmissionLineTransient::CreateOperatorA6()
{
   A6 = new SparseMatrix*[nbrGroups];
   stringstream ss; 
   char fn[256];
   for(int gc = 0; gc < nbrGroups; gc++)
   {
      ConstantCoefficient One(1.0);
      real_t wireArea = IntegrateScalar(*Afespace, One, gc+1);
      A6[gc] = new SparseMatrix(1, 1);
      A6[gc]->Set(0, 0, sigma_ * wireArea);
      A6[gc]->Finalize();

      sprintf(fn,"out/A6_%d.txt", gc );
      std::ofstream out(fn);
      if(printMatrix) A6[gc]->Print(out, 10);

      cout << A6[gc]->Height() << " A6 Height()\n " 
           << A6[gc]->Width()  << " A6 Width()\n\n ";
   }
   return 1;
}


int TransmissionLineTransient::CreateOperatorA7()
{
   A7 = new SparseMatrix*[nbrGroups];

   stringstream ss; 
   char fn[256];
   for(int gc = 0; gc < nbrGroups; gc++)
   {
      A7[gc] = new SparseMatrix(*(A5[gc]));
      *(A7[gc]) *= -1.0;
      A7[gc]->Finalize();

      sprintf(fn,"out/A7_%d.txt", gc );        
      std::ofstream out(ss.str());
      if(printMatrix) A7[gc]->Print(out, 10);

      cout << A7[gc]->Height() << " A7 Height()\n " 
         << A7[gc]->Width()  << " A7 Width()\n\n ";
   }
   return 1;
}


int TransmissionLineTransient::CreaterhsVector()
{
// computing rhs 
   rhs = new Vector(2*nbrdof+2*nbrGroups);
   *rhs = 0.0;
   for(int gc=0; gc < nbrGroups; gc++)
   {
      (*rhs)[2*nbrdof + 2*gc + 0] = groupsInfo[gc].current[0] * cos(2*M_PI*groupsInfo[gc].current[1]/360.0);
      (*rhs)[2*nbrdof + 2*gc + 1] = groupsInfo[gc].current[0] * sin(2*M_PI*groupsInfo[gc].current[1]/360.0);;     
   }

   std::ofstream out("out/rhs.txt");
   if(printMatrix) rhs->Print(out, 10);

   cout << rhs->Size() << " rhs size\n\n ";
   return 1;

}

int TransmissionLineTransient::CreatexVector()
{
   x = new Vector(2*nbrdof+2*nbrGroups);
   *x = 0.0; 
   std::ofstream out("out/x.txt");
   if(printMatrix) x->Print(out, 10);
   cout << x->Size() << " x size\n\n ";
   return 1;
}

int TransmissionLineTransient::CreateBlockOperator()
{

// 6. Define the BlockStructure of the problem
   int blockoffsetsize = 2 + 2 * nbrGroups + 1;
   blockOffset = new Array<int>(blockoffsetsize);  //n+1
   (*blockOffset)[0]=0;
   (*blockOffset)[1]=nbrdof; 
   (*blockOffset)[2]=nbrdof;
   for(int i = 3; i<blockoffsetsize; i++) (*blockOffset)[i]=1;
   blockOffset->PartialSum();
   {
      std::ofstream out("out/blockOffset.txt");
      blockOffset->Print(out, 10);
   }
  
   Device device("cpu");
   MemoryType mt = device.GetMemoryType();

   ProxOp = new BlockOperator(*blockOffset);
   
// Build the operator, insert each block.
// row 0 ...
      ProxOp->SetBlock(0, 0, A1);
      ProxOp->SetBlock(0, 1, A2);
      for(int i = 0; i<nbrGroups; i++) ProxOp->SetBlock(0, 2+2*i, A3[i]);
// row 1
      ProxOp->SetBlock(1, 0, A4);
      ProxOp->SetBlock(1, 1, A1);
      for(int i = 0; i<nbrGroups; i++) ProxOp->SetBlock(1, 3+2*i, A3[i]);
// col 0
      for(int i = 0; i<nbrGroups; i++) ProxOp->SetBlock(3+2*i, 0, A7[i]);
// col 1
      for(int i = 0; i<nbrGroups; i++) ProxOp->SetBlock(2+2*i, 1, A5[i]);
// diagonal (2, 2)
      for(int i = 0; i<nbrGroups; i++)
      {
         ProxOp->SetBlock(2+2*i, 2+2*i, A6[i]);
         ProxOp->SetBlock(3+2*i, 3+2*i, A6[i]);
      }

      {
      std::ofstream out("out/ProxOp.txt");
      if(printMatrix) ProxOp->PrintMatlab(out);
      }

      assert(ProxOp->Height() == 2*nbrdof+2*nbrGroups);
      assert(ProxOp->Width() == 2*nbrdof+2*nbrGroups);
      
//DL241125: I check ProxOp it contains all the BLFA1 to 4 in proper order.

      assert(2*A1->NumRows()+2*nbrGroups==ProxOp->NumRows());
      assert(2*A1->NumCols()+2*nbrGroups==ProxOp->NumCols());


   A = new BlockOperator(*blockOffset);
   A_ptr = A;

   B = new BlockVector(*blockOffset, mt);
   X = new BlockVector(*blockOffset, mt);

   // note the A_ptr do not point at the same operator after !!!
   ProxOp->FormLinearSystem(*ess_tdof_list_block, *x, *rhs, A_ptr, *X, *B);

   {
      std::ofstream out("out/X.txt");
      if(printMatrix) X->Print(out, 10);
   }

   {
      std::ofstream out("out/B.txt");
      if(printMatrix) B->Print(out, 10);
   }

   {
      std::ofstream out("out/A_ptr.txt");
      if(printMatrix) A_ptr->PrintMatlab(out);
   }

   cout << A_ptr->Height() << " Operator Height()\n " 
         << A_ptr->Width()  << " Operator Width()\n "
         << X->Size() << " X.Size()\n " 
         << B->Size() << " B.Size()\n\n ";

   return 1;
}
   
int TransmissionLineTransient::CreatePreconditionner()
{

   // 10. Construct the operators for preconditioner

   // ************************************
   // Here I have no idea how to build a preconditionner.
   // ex5 is not appropriated as per my understanding.
   // ************************************

   // Create smoothers for diagonal blocks
   gs = new GSSmoother*[2+2*nbrGroups];
   int gc;
   gs[0] = new GSSmoother(*A1); // Gauss-Seidel smoother for A11
   gs[1] = new GSSmoother(*A1); // Gauss-Seidel smoother for A22

   for(gc=0;gc<nbrGroups; gc++)
   {
      gs[2*gc+2] = new GSSmoother(*(A6[gc]));
      gs[2*gc+3] = new GSSmoother(*(A6[gc]));
   }

   block_prec = new BlockDiagonalPreconditioner(*blockOffset);
   block_prec->SetDiagonalBlock(0, gs[0]); // Set smoother for A11
   block_prec->SetDiagonalBlock(1, gs[1]); // Set smoother for A22
   for(gc=0;gc<nbrGroups; gc++)
   {
      block_prec->SetDiagonalBlock(2+2*gc+0, gs[2*gc+2]);
      block_prec->SetDiagonalBlock(2+2*gc+1, gs[2*gc+3]);
   }
   return 1;
}

int TransmissionLineTransient::Solver()
{
   // Solve system Ax = b
   GMRESSolver solver;
   solver.SetOperator(*A_ptr);
   solver.SetPreconditioner(*block_prec);
   solver.SetRelTol(1e-12);
   //   solver.SetAbsTol(1e-8);
   solver.SetMaxIter(50000);
   solver.SetPrintLevel(1);

   solver.Mult(*B, *X);

   A->RecoverFEMSolution(*X, *rhs, *x);
   {
      std::ofstream out("out/xsol.txt");
      if(printMatrix) x->Print(out, 10);
   }
   return 1;
}


int TransmissionLineTransient::PostPrecessing()
{
   AzrGF = new GridFunction(Afespace);
   AziGF = new GridFunction(Afespace);
  
   // rebuild GFR and GFI from x.
   AzrGF->MakeRef(Afespace, *x, 0);
   AziGF->MakeRef(Afespace, *x, nbrdof);

/* B-field never went well...
   // compute the magnetic field, curl(A).

   Bfec = new ND_FECollection(order, dim);
   Bfespace = new FiniteElementSpace(mesh, Bfec);
   
   BrGF = new GridFunction(Bfespace);
   BiGF = new GridFunction(Bfespace);

   MixedBilinearForm MBLF(Afespace, Bfespace);
   MBLF.AddDomainIntegrator(new MixedScalarWeakCurlIntegrator);
   MBLF.Assemble();
   MBLF.Finalize();

   MBLF.Mult(*AzrGF, *BrGF);
   MBLF.Mult(*AziGF, *BiGF);

   VectorGridFunctionCoefficient BrCoeff(BrGF);
   VectorGridFunctionCoefficient BiCoeff(BiGF);

   InnerProductCoefficient BrnCoeff(BrCoeff, BrCoeff);
   InnerProductCoefficient BinCoeff(BiCoeff, BiCoeff);
   SumCoefficient BSquareCoeff(BrnCoeff, BinCoeff);
   PowerCoefficient BCoeff(BSquareCoeff, 0.5);

   BNormfec = new H1_FECollection(order, dim);
   BNormfespace = new FiniteElementSpace(mesh, BNormfec);

   BnGF = new GridFunction(BNormfespace);
   BnGF->ProjectCoefficient(BCoeff);
   
   Glvis(mesh, BrGF, "B-Real", 8, "mvve" );
   Glvis(mesh, BiGF, "B-Imag", 8, "mvve" );
   Glvis(mesh, BnGF, "B", 8, "mvve" );
*/
   // compute Jr 

   GridFunctionCoefficient AzrGFCoeff(AzrGF);
   GridFunctionCoefficient AziGFCoeff(AziGF);

   Vector CoeffVector(2+nbrGroups);
   CoeffVector = 0.0;
   for(int gc=0; gc<nbrGroups; gc++) CoeffVector[gc] = omega_*sigma_;
   PWConstCoefficient K1(CoeffVector);
   
   ProductCoefficient Jr1Coeff(K1, AziGFCoeff);

   PWConstCoefficient K2;
   
   PWConstCoefficient Jr2Coeff;
   Vector CoeffVector2(2+nbrGroups);
   CoeffVector2 = 0.0;
   for(int gc=0; gc<nbrGroups; gc++) CoeffVector2[gc] = sigma_*(*x)[2*nbrdof+gc*2];
   Jr2Coeff.UpdateConstants(CoeffVector2);
  
   SumCoefficient JrCoeff(Jr1Coeff, Jr2Coeff);


   // Compute Ji
   PWConstCoefficient K3;   
   Vector CoeffVector3(2+nbrGroups);
   CoeffVector3 = 0.0;
   for(int gc=0; gc<nbrGroups; gc++) CoeffVector3[gc] = -omega_*sigma_;
   K3.UpdateConstants(CoeffVector3);
   
   ProductCoefficient Ji1Coeff(K3, AzrGFCoeff);

   PWConstCoefficient Ji2Coeff;
   Vector CoeffVector4(2+nbrGroups);
   CoeffVector4 = 0.0;
   for(int gc=0; gc<nbrGroups; gc++) CoeffVector4[gc] = sigma_*(*x)[2*nbrdof+2*gc+1];
   Ji2Coeff.UpdateConstants(CoeffVector4);
   
   SumCoefficient JiCoeff(Ji1Coeff, Ji2Coeff);

   //Compute J
   PowerCoefficient jrSquareCoeff(JrCoeff, 2.0);
   PowerCoefficient jiSquareCoeff(JiCoeff, 2.0);
   SumCoefficient JSquare(jrSquareCoeff, jiSquareCoeff);
   PowerCoefficient JCoeff(JSquare, 0.5);

   JFec = new DG_FECollection(order, dim);
   JFESpace = new FiniteElementSpace(mesh, JFec);

   JiGF = new GridFunction(JFESpace);
   JiGF->ProjectCoefficient(JiCoeff);
   JrGF = new GridFunction(JFESpace);
   JrGF->ProjectCoefficient(JrCoeff);
   JGF = new GridFunction(JFESpace);
   JGF->ProjectCoefficient(JCoeff);

   return 1;
}

int TransmissionLineTransient::DisplayResults()
{

   Glvis(mesh, AzrGF, "A-Real" );
   Glvis(mesh, AziGF, "A-Imag" );

   Glvis(mesh, JrGF, "J-Real" );
   Glvis(mesh, JiGF, "J-Imag" );
   Glvis(mesh, JGF, "J" );

   ConstantCoefficient One(1.0);
   
   cout << "\n";
   for(int gc=0; gc<nbrGroups; gc++)
   {
      cout << "wire " << gc << ": "<< (*x)[2*nbrdof+2*gc] << " Vr, "
           << (*x)[2*nbrdof+2*gc+1] << " Vi, "
           << sqrt(pow((*x)[2*nbrdof+2*gc], 2)+pow((*x)[2*nbrdof+2*gc+1], 2)) << " V\n";
   
   real_t WireArea = IntegrateScalar(*Afespace, One, gc+1);
   real_t Rdc = 1.0/(WireArea * sigma_);
   real_t Rac = (*x)[2*nbrdof+2*gc] / (*rhs)[2*nbrdof+2*gc] ;
   real_t RacdcRatio = Rac/Rdc;
   real_t Lw = (*x)[2*nbrdof+2*gc+1] / ((*rhs)[2*nbrdof+2*gc]  * omega_);

   cout << Rdc << " Rdc\n"
        << Rac << " Rac\n"
        << RacdcRatio << " AC DC Ratio at " << omega_/2.0/M_PI << "Hz\n"
        << 1E6*Lw << " L uH\n\n";
   }
   return 1;
}

int TransmissionLineTransient::CleanOutDir()
{
    system("rm -f out/*");
    return 1;
}

int TransmissionLineTransient::Save()
{
   char fileName[250];
   
   // 14. Save data in the VisIt format
   sprintf(fileName, "visit/%s", meshFile);
   VisItDataCollection visit_dc(fileName, mesh);
   visit_dc.RegisterField("AzReal",AzrGF);
   visit_dc.RegisterField("AzImag",AziGF);
   visit_dc.RegisterField("JReal",JrGF);
   visit_dc.RegisterField("JImag",JiGF);
   visit_dc.RegisterField("J",JGF);
   visit_dc.Save();

   // 15. Save data in the ParaView format
   
   sprintf(fileName, "paraview/%s", meshFile);
   ParaViewDataCollection paraview_dc(fileName, mesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("AzReal",AzrGF);
   paraview_dc.RegisterField("AzImag",AziGF);
   paraview_dc.RegisterField("JReal",JrGF);
   paraview_dc.RegisterField("JImag",JiGF);
   paraview_dc.RegisterField("J",JGF);
   paraview_dc.Save();
   return 1;
}


int main(int argc, char *argv[])
{

   TransmissionLineTransient PE;
   TwoDWiresGenerator WG;

   WG.Parser(argc, argv);
   WG.ReadConfigFile();

   PE.SetNbrGroups(WG.GetNbrGroups());
   PE.SetGroupsInfo(WG.GetGroupsInfo());

   PE.Parser(argc, argv);
   PE.CleanOutDir();
   PE.LoadMeshFile();
   PE.CreateFESpace();
   PE.CreateEssentialBoundary();
   PE.CreateOperatorA1();
   PE.CreateOperatorA2();
   PE.CreateOperatorA3();
   PE.CreateOperatorA4();
   PE.CreateOperatorA5();
   PE.CreateOperatorA6();
   PE.CreateOperatorA7();
   PE.CreaterhsVector();
   PE.CreatexVector();
   PE.CreateBlockOperator();
   PE.CreatePreconditionner();
   PE.Solver();
   PE.PostPrecessing();
   PE.DisplayResults();
   PE.Save();
   return 0;
}
