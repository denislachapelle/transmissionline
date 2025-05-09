//                                se2dblock.cpp
//                                inspired from proximity2dblockb
//                                inspired from proximity2dblock
//                                inspired from proximity2d
//                                based on MFEM Example 22 prob 1 (case 0), ex5...
//
// Compile with: make se2dblock, need MFEM version 4.7 and GLVIS-4.3.
//
// Sample runs:  ./se2dblock
//
/*
Description:  

Implementation suggested by paul hilsher in https://github.com/mfem/mfem/issues/4584
from paper "Specialized conductor models for finite element eddy current simulation".
https://www.iem.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaabcfymi

The equation is 
- div(grad(Az)) + i w u s Az - u s V = 0

Az : magnetic vector potential z-component.
V : voltage potential.
u : permeability.
s : conductivity.
w : pulsation, 2 pi f.
Assume 1m long, so s V is electric field.

then split in real and imag parts
- div(grad(Azr)) - w u s Azi - u s Vr = 0
- div(grad(Azi)) + w u s Azr - u s Vi = 0

Let define matrix A1, A2, A3 and A4 as implementing ....
A1 -div(grad((.))
A2 - u w s (.)
A3 - u s (.)
A4 w u s (.)

We also want to enforce the current ..
intS(- i s w A) + V/R = I

then split real and imaginary equation...
intS(s w Ai) + Vr/R = Ir
intS(- s w Ar) + Vi/R = Ii

let define A5 and A6 as implementing ...
A5 intS(s w (.))
A6 1/R (.)
A7 intS(-s w (.))

Then we can write the assembled matrix...

[A1 A2 A3 0 ] [Azr] = [0]
[A4 A1 0  A3] [Azi] = [0]
[0  A5 A6 0 ] [Vr]  = [Ir]
[A7 0  0  A6] [Vi]  = [Ii]

[dofxdof dofxdof dofx1 dofx1] [dofx1] = [dofx1]
[dofxdof dofxdof dofx1 dofx1] [dofx1] = [dofx1]
[1xdof   1xdof   1x1   1x1]   [dofx1] = [dofx1]
[1xdof   1xdof   1x1   1x1]   [dofx1] = [dofx1]

Ir being total real current in wire 1.
Ii being total imaginary current in wire 1.

Once solved the current density can be computed...
J = - i w s A - s V

Jr =   w s Ai + s Vr
Ji = - w s Ar + s Vi

||J|| = sqrt(Jr^2+Ji^2)

u: permeability.
e: permitivity.
s: conductivity.
i: sqrt(-1)
*/

#include <mfem.hpp>
#include <linalg/hypre.hpp>
#include "mytools.hpp"
#include <iostream>
#include <math.h>
#include <filesystem>


#define wire_1 1
#define wire1contour 5

using namespace std;
using namespace mfem;

class ProximityEffect
{
   private:
   // 1. Parse command-line options default.
   const char *mesh_file = "oneroundwireonly2d.msh";
   int order = 1;
   double freq = -1.0;
   real_t Iw1r = 1.0;
   real_t Iw1i = 0.0;
   real_t mu_ = 1.257E-6;
   real_t epsilon_ = 8.854E-12;
   real_t sigma_ = 58E6;
   real_t omega_ = 2 * M_PI * 60;

   Mesh *mesh;
   int dim;
   int nbrel;  //number of element.

   FiniteElementCollection *fec;
   FiniteElementSpace *fespace;
   int nbrdof;   //number of degree of freedom.

   Array<int> *ess_tdof_list_block; //essential dof list.

   //matrix used to form the block operator.
   SparseMatrix *A1, *A2, *A3, *A4, *A5, *A6, *A7;
   Vector *rhs, *x;

   BlockOperator *A;
   Array<int> *blockOffset;
   BlockOperator *ProxOp;

   mfem::Operator *A_ptr;
   BlockVector *B, *X;
   
   BlockDiagonalPreconditioner *block_prec;

   GridFunction *AzrGF, *AziGF; // magnetic vector potential z-axis, real and imaginary.

   //Space and gridfunction for the current density.
   FiniteElementCollection *JFec;
   FiniteElementSpace *JFESpace;

   GridFunction *JrGF, *JiGF, *JGF;


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

};

int ProximityEffect::Parser(int argc, char *argv[])
{

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&mu_, "-mu", "--permeability",
                  "Permeability of free space (or 1/(spring constant)).");
   args.AddOption(&epsilon_, "-eps", "--permittivity",
                  "Permittivity of free space (or mass constant).");
   args.AddOption(&sigma_, "-sigma", "--conductivity",
                  "Conductivity (or damping constant).");
   args.AddOption(&freq, "-f", "--frequency",
                  "Frequency (in Hz).");
   args.AddOption(&Iw1r, "-iw1r", "--iw1r",
                  "Real current in wire #1.");
   args.AddOption(&Iw1i, "-iw1i", "--iw1i",
                  "Imaginary current in wire #1.");
                
  
   args.Parse();

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device("cpu");
   device.Print();

   if ( !(freq < 0.0) ) omega_ = 2.0 * M_PI * freq;

   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   return 0;
}

int ProximityEffect::LoadMeshFile()
{

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes
   //    with the same code.
   mesh = new Mesh(mesh_file, 1, 1); // 
   dim = mesh->Dimension();

   cout << mesh->Dimension() << " dimensions\n"
        << mesh->GetNV() << " vertices\n"
        << mesh->GetNE() << " elements\n"
        << mesh->GetNBE() << " boundary elements\n"
        << mesh->GetNEdges() << " edges\n";

   nbrel = mesh->GetNE();

   return 1;
}

int ProximityEffect::CreateFESpace()
{
   fec = new H1_FECollection(order, dim);
   fespace = new FiniteElementSpace(mesh, fec);
   nbrdof = fespace->GetNDofs(); 
   cout << fespace->GetNDofs() << " degree of freedom\n"
        << fespace->GetVDim() << " vectors dimension\n\n";   

return 1;
}

int ProximityEffect::CreateEssentialBoundary()
{

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined based on the type
   //    of mesh and the problem type.

   // real and imag are the same because they refer to mesh nodes.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   assert(mesh->bdr_attributes.Max()==5);
   ess_bdr.SetSize(mesh->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[wire1contour-1]=1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   {
      std::ofstream out("out/ess_tdof_list_R.txt");
      ess_tdof_list.Print(out, 10);
   }

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

int ProximityEffect::CreateOperatorA1()
{
   ConstantCoefficient One(1.0);

   BilinearForm BLFA1(fespace);
   BLFA1.AddDomainIntegrator(new DiffusionIntegrator(One));
   BLFA1.Assemble();
   BLFA1.Finalize();
   
   A1 = new SparseMatrix(BLFA1.SpMat());

   std::ofstream out("out/A1.txt");
   A1->Print(out, 10);
  
   cout << A1->Height() << " A1 Height()\n " 
        << A1->Width()  << " A1 Width()\n\n ";

   return 1;
}

int ProximityEffect::CreateOperatorA2()
{
   
   ConstantCoefficient K(-mu_*omega_*sigma_);
   
   BilinearForm BLFA2(fespace);
   BLFA2.AddDomainIntegrator(new MassIntegrator(K));
   BLFA2.Assemble();
   BLFA2.Finalize();
   
   A2 = new SparseMatrix(BLFA2.SpMat());

   std::ofstream out("out/A2.txt");
   A2->Print(out, 10);
  
   cout << A2->Height() << " A2 Height()\n " 
        << A2->Width()  << " A2 Width()\n\n ";

   return 1;
}

int ProximityEffect::CreateOperatorA3()
{  

   ConstantCoefficient K(-mu_*sigma_);
   
   BilinearForm BLFA3(fespace);
   BLFA3.AddDomainIntegrator(new MassIntegrator(K));
   BLFA3.Assemble();
   BLFA3.Finalize();

   std::ofstream out4("out/BLFA3.txt");
   BLFA3.SpMat().Print(out4, 10);
   
   SparseMatrix TempSM(BLFA3.SpMat());
   TempSM.Finalize();

   std::ofstream out1("out/TempSM.txt");
   TempSM.Print(out1, 10);
   
   Vector TempVEC(nbrdof);
   TempSM.GetRowSums(TempVEC);

   std::ofstream out2("out/TempVEC.txt");
   TempVEC.Print(out2, 10);

   A3 = new SparseMatrix(nbrdof, 1);
   *A3 = 0.0;
   
   for(int i=0; i<nbrdof; i++)
   {
      A3->Add(i, 0, TempVEC[i]);
   }

   A3->Finalize();

   std::ofstream out3("out/A3.txt");
   A3->Print(out3, 10);
   
   cout << A3->Height() << " A3 Height()\n " 
        << A3->Width()  << " A3 Width()\n\n ";

   return 1;
}


int ProximityEffect::CreateOperatorA4()
{
   A4 = new SparseMatrix(*A2);
   *A4 *= -1.0;
   A4->Finalize();
      
   std::ofstream out("out/A4.txt");
   A4->Print(out, 10);

   cout << A4->Height() << " A4 Height()\n " 
        << A4->Width()  << " A4 Width()\n\n ";

   return 1;
}

int ProximityEffect::CreateOperatorA5()
{
   /*
This section of code compute the operator performing the 
integration.

For each element compute the current which is the integral of J x s.
Note s, the conductivity, is a PWCoefficient
*/
 
   ConstantCoefficient K(omega_*sigma_);
 
   //A5 method #2, linearform.
   //surface integral.
   LinearForm LFA5(fespace);
   LFA5.AddDomainIntegrator(new DomainLFIntegrator(K));
   LFA5.Assemble();
   
   std::ofstream out1("out/LFA5.txt");
   LFA5.Print(out1, 10);

   A5 = new SparseMatrix(1, nbrdof);
   for(int k=0; k<nbrdof; k++)
   {
      A5->Set(0, k, LFA5[k]);
   }

   A5->Finalize();

   std::ofstream out2("out/A5.txt");
   A5->Print(out2, 10);

   cout << A5->Height() << " A5 Height()\n " 
        << A5->Width()  << " A5 Width()\n\n ";

   return 1;
}


int ProximityEffect::CreateOperatorA6()
{

   ConstantCoefficient One(1.0);
   real_t WireArea = IntegrateScalar(*fespace, One, wire_1);
   A6 = new SparseMatrix(1, 1);
   A6->Set(0, 0, sigma_ * WireArea);
   A6->Finalize();

   std::ofstream out("out/A6.txt");
   A6->Print(out, 10);

   cout << A6->Height() << " A6 Height()\n " 
        << A6->Width()  << " A6 Width()\n\n ";

   return 1;
}


int ProximityEffect::CreateOperatorA7()
{
   A7 = new SparseMatrix(*A5);
   *A7 *= -1.0;
   A7->Finalize();
      
   std::ofstream out("out/A7.txt");
   A7->Print(out, 10);

   cout << A7->Height() << " A7 Height()\n " 
        << A7->Width()  << " A7 Width()\n\n ";

   return 1;
}


int ProximityEffect::CreaterhsVector()
{
// computing rhs 
   rhs = new Vector(2*nbrdof+2);
   *rhs = 0.0;
   (*rhs)[2*nbrdof + 0] = Iw1r;
   (*rhs)[2*nbrdof + 1] = Iw1i;
   std::ofstream out("out/rhs.txt");
   rhs->Print(out, 10);

   cout << rhs->Size() << " rhs size\n\n ";
   return 1;

}

int ProximityEffect::CreatexVector()
{
   x = new Vector(2*nbrdof+2);
   *x = 0.0; 
   std::ofstream out("out/x.txt");
   x->Print(out, 10);
   cout << x->Size() << " x size\n\n ";
   return 1;
}

int ProximityEffect::CreateBlockOperator()
{

// 6. Define the BlockStructure of the problem.
   blockOffset = new Array<int>(5);  //n+1
   (*blockOffset)[0]=0;
   (*blockOffset)[1]=A1->NumRows(); 
   (*blockOffset)[2]=A4->NumRows();
   (*blockOffset)[3]=A5->NumRows();
   (*blockOffset)[4]=A7->NumRows();
    blockOffset->PartialSum();
   {
      std::ofstream out("out/blockOffset.txt");
      blockOffset->Print(out, 10);
   }
  
   // 8. Allocate memory (x, rhs) for the analytical solution and the right hand
   //    side.  Define the GridFunction u,p for the finite element solution and
   //    linear forms fform and gform for the right hand side.  The data
   //    allocated by x and rhs are passed as a reference to the grid functions
   //    (u,p) and the linear forms (fform, gform).
  
  Device device("cpu");
  MemoryType mt = device.GetMemoryType();

   ProxOp = new BlockOperator(*blockOffset);

   
// Build the operator row by row...
      ProxOp->SetBlock(0, 0, A1);
      ProxOp->SetBlock(0, 1, A2);
      ProxOp->SetBlock(0, 2, A3);

      ProxOp->SetBlock(1, 0, A4);
      ProxOp->SetBlock(1, 1, A1);
      ProxOp->SetBlock(1, 3, A3);
      
      ProxOp->SetBlock(2, 1, A5);
      ProxOp->SetBlock(2, 2, A6);

      ProxOp->SetBlock(3, 0, A7);
      ProxOp->SetBlock(3, 3, A6);

      {
      std::ofstream out("out/ProxOp.txt");
      ProxOp->PrintMatlab(out);
      }

      assert(ProxOp->Height() == 2*nbrdof+2);
      assert(ProxOp->Width() == 2*nbrdof+2);
      
//DL241125: I check ProxOp it contains all the BLFA1 to 4 in proper order.

      assert(2*A1->NumRows()+2==ProxOp->NumRows());
      assert(2*A1->NumCols()+2==ProxOp->NumCols());


   A = new BlockOperator(*blockOffset);
   A_ptr = A;

   B = new BlockVector(*blockOffset, mt);
   X = new BlockVector(*blockOffset, mt);

   // note the A_ptr do not point at the same operator after !!!
   ProxOp->FormLinearSystem(*ess_tdof_list_block, *x, *rhs, A_ptr, *X, *B);

   {
      std::ofstream out("out/X.txt");
      X->Print(out, 10);
   }

   {
      std::ofstream out("out/B.txt");
      B->Print(out, 10);
   }

   {
      std::ofstream out("out/A_ptr.txt");
      A_ptr->PrintMatlab(out);
   }

   cout << A_ptr->Height() << " Operator Height()\n " 
         << A_ptr->Width()  << " Operator Width()\n "
         << X->Size() << " X.Size()\n " 
         << B->Size() << " B.Size()\n\n ";

   return 1;
}
   
int ProximityEffect::CreatePreconditionner()
{

   // 10. Construct the operators for preconditioner

// ************************************
// Here I have no idea how to build a preconditionner.
// ex5 is not appropriated as per my understanding.
// ************************************

// Create smoothers for diagonal blocks
GSSmoother gs1(*A1); // Gauss-Seidel smoother for A11
GSSmoother gs2(*A1); // Gauss-Seidel smoother for A22
GSSmoother gs3(*A6);
GSSmoother gs4(*A6);

block_prec = new BlockDiagonalPreconditioner(*blockOffset);
block_prec->SetDiagonalBlock(0, &gs1); // Set smoother for A11
block_prec->SetDiagonalBlock(1, &gs2); // Set smoother for A22
block_prec->SetDiagonalBlock(2, &gs3);
block_prec->SetDiagonalBlock(3, &gs4);

return 1;
}

int ProximityEffect::Solver()
{
   // Solve system Ax = b
   GMRESSolver solver1;
   solver1.SetOperator(*A_ptr);
 //solver1.SetPreconditioner(*block_prec);
   solver1.SetRelTol(1e-16);
   //   solver.SetAbsTol(1e-8);
   solver1.SetMaxIter(100000);
   solver1.SetPrintLevel(1);

   //x = 0.0;       // Initial guess
   solver1.Mult(*B, *X);
  

   A->RecoverFEMSolution(*X, *rhs, *x);
   {
      std::ofstream out("out/xsol.txt");
      x->Print(out, 10);
   }

   return 1;
   
}


int ProximityEffect::PostPrecessing()
{
   AzrGF = new GridFunction(fespace);
   AziGF = new GridFunction(fespace);
   
   // rebuild GFR and GFI from x.
   AzrGF->MakeRef(fespace, *x, 0);
   AziGF->MakeRef(fespace, *x, nbrdof);

   real_t Vr = (*x)[2*nbrdof+0];
   real_t Vi = (*x)[2*nbrdof+1];

   GridFunctionCoefficient AzrGFCoeff(AzrGF);
   GridFunctionCoefficient AziGFCoeff(AziGF);

   // compute Jr   
   ConstantCoefficient K1(omega_*sigma_);
   ProductCoefficient Jr1Coeff(K1, AziGFCoeff);

   
   ConstantCoefficient K2(sigma_);
   
   ProductCoefficient Jr2Coeff(Vr, K2);

   SumCoefficient JrCoeff(Jr1Coeff, Jr2Coeff);

   // Compute Ji
   ConstantCoefficient K3(-omega_*sigma_);
   ProductCoefficient Ji1Coeff(K3, AzrGFCoeff);

   
   ProductCoefficient Ji2Coeff(Vi, K2);

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

int ProximityEffect::DisplayResults()
{

   Glvis(mesh, AzrGF, "A-Real" );
   Glvis(mesh, AziGF, "A-Imag" );

   Glvis(mesh, JrGF, "J-Real" );
   Glvis(mesh, JiGF, "J-Imag" );
   Glvis(mesh, JGF, "J" );

   cout << (*x)[2*nbrdof+0] << " Vr\n"
        << (*x)[2*nbrdof+1] << " Vi\n"
        << sqrt(pow((*x)[2*nbrdof+0], 2)+pow((*x)[2*nbrdof+1], 2)) << " V\n";
   
   ConstantCoefficient One(1.0);
   real_t WireArea = IntegrateScalar(*fespace, One, wire_1);
   real_t Rdc = 1.0/(WireArea * sigma_);
   real_t Rac = (*x)[2*nbrdof+0] / Iw1r;
   real_t RacdcRatio = Rac/Rdc;
   real_t Lw = (*x)[2*nbrdof+1] / (Iw1r * omega_);


   cout << Rdc << " Rdc\n"
        << Rac << " Rac\n"
        << RacdcRatio << " AC DC Ratio at " << omega_/2.0/M_PI << "Hz\n"
        << 1E6*Lw << " L uH\n";
   

return 1;
}

int ProximityEffect::CleanOutDir()
{
    system("rm -f out/*");
    return 1;
}



int main(int argc, char *argv[])
{

   StopWatch chrono;
   tic();
   
   ProximityEffect PE;

   PE.CleanOutDir();
   
   PE.Parser(argc, argv);

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

   
   return 0;
}
