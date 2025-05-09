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
#include "mytools.hpp"
#include <iostream>
#include <math.h>
#include <filesystem>

//Physical group expected from mesh file.
#define tl1 1
#define tl2 2
#define input 3
#define output 4
#define stubend 5

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
      FiniteElementSpace *VFESpace, *IFESpace;
      int Vnbrdof, Inbrdof;   //number of degree of freedom.

       BilinearForm *M_V, *M_I;
       MixedBilinearForm *K_VI, *K_IV;

      //matrix pointer and matrix array pointer
      Vector *rhs, *x;

      Operator *A_ptr;
      BlockVector *B, *X;
      
      GSSmoother **gs;
      BlockDiagonalPreconditioner *block_prec;

      GridFunction *VGF, *IGF; 
      
   private:
      real_t Source(real_t t);   

   public:
      //delete all files in out dir.
      int CleanOutDir();
      //parse the options.
      int Parser(int argc, char *argv[]);
      int LoadMeshFile();
      int CreateFESpace();
      int CreateEssentialBoundary();
      int CreateBilinears();
      int TimeSteps();
      int CreaterhsVector();
      int CreatexVector();
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
   assert(dim==1); //this software works only for dimension 1.

   cout << mesh->bdr_attributes.Max() << " bdr attr max\n"
        << mesh->Dimension() << " dimensions\n"
        << mesh->SpaceDimension() << " space dimensions\n"
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
   Vnbrdof = VFESpace->GetNDofs(); 
   cout << "VFESpace\n" << VFESpace->GetNDofs() << " degree of freedom\n"
        << VFESpace->GetVDim() << " vectors dimension\n\n";   

   IFEC = new L2_FECollection(order-1, dim);
   IFESpace = new FiniteElementSpace(mesh, IFEC);
   Inbrdof = IFESpace->GetNDofs(); 
   cout << "IFESpace\n" << IFESpace->GetNDofs() << " degree of freedom\n"
        << IFESpace->GetVDim() << " vectors dimension\n\n";   

return 1;
}

int TransmissionLineTransient::CreateEssentialBoundary()
{
   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   assert(mesh->bdr_attributes.Max()==5);
   ess_bdr.SetSize(mesh->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[output-1]=1;
   ess_bdr[stubend-1]=1;
   VFESpace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   {
      std::ofstream out("out/V_ess_tdof_list.txt");
      ess_tdof_list.Print(out, 10);
   }

   ess_bdr = 0;
   IFESpace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   {
      std::ofstream out("out/I_ess_tdof_list.txt");
      ess_tdof_list.Print(out, 10);
   }

   return 1;
}

int TransmissionLineTransient::CreateBilinears()
{
   // Set up the mass and stiffness matrices
   // Constants for the telegrapher’s equation
   double L = 1.0e-6;  // Inductance per unit length
   double C = 100.0e-12; // Capacitance per unit length
   double R = 100.0;  // Resistance per unit length
   double G = 1.0e-6;  // Conductance per unit length

   M_V = new BilinearForm(VFESpace);
   M_I = new BilinearForm(IFESpace);
   K_IV = new MixedBilinearForm(IFESpace, VFESpace);
   K_VI = new MixedBilinearForm(VFESpace, IFESpace);

   // Define the mass and stiffnes
   ConstantCoefficient a(-G/C);
    
   Vector vectorB(2); vectorB[0]=-1.0/C; vectorB[1]=0.0;
   VectorConstantCoefficient b(vectorB); 
    
   ConstantCoefficient c(-R/L); 

   Vector vectorD(2); vectorD[0]=-1.0/L; vectorD[1]=0.0;
   VectorConstantCoefficient d(vectorD); 
       
   M_V->AddDomainIntegrator(new MassIntegrator(a));
   K_IV->AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(b));
   M_I->AddDomainIntegrator(new MassIntegrator(c));
   K_VI->AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(d));
    
   M_V->Assemble();
   M_I->Assemble();
   K_VI->Assemble();
   K_IV->Assemble();

   M_V->Finalize();
   M_I->Finalize();
   K_VI->Finalize();
   K_IV->Finalize();

   cout << M_V->Height() << "M_V->Height()\n";
   cout << M_V->Width() << "M_V->Width()\n";
   cout << M_I->Height() << "M_V->Height()\n";
   cout << M_I->Width() << "M_V->Width()\n";
   cout << K_IV->Height() << "K_IV->Height()\n";
   cout << K_IV->Width() << "K_IV->Width()\n";
   cout << K_VI->Height() << "K_VI->Height()\n";
   cout << K_VI->Width() << "K_VI->Width()\n";
   

   {
      std::ofstream out("out/M_V.txt");
      if(printMatrix) M_V->SpMat().Print(out, 10);
   }

   {
      std::ofstream out("out/M_I.txt");
      if(printMatrix) M_I->SpMat().Print(out, 10);
   }

   {
      std::ofstream out("out/K_VI.txt");
      if(printMatrix) K_VI->SpMat().Print(out, 10);
   }

   {
      std::ofstream out("out/K_IV.txt");
      if(printMatrix) K_IV->SpMat().Print(out, 10);
   }

   return 1;
}

real_t TransmissionLineTransient::Source(real_t t)
{
   if(t==0) return 0.0;
   else return 1.0;
}

int TransmissionLineTransient::TimeSteps()
{
   // Initialize solutions for V and I (voltage and current)
   GridFunction VGF(VFESpace);
   GridFunction IGF(IFESpace);
   VGF = 0.0;
   IGF = 0.0;
   
   // Set up time-stepping parameters
   double t = 0.0;       // Initial time
   double t_final = 1800e-9; // Final time
   double dt = (t_final - t)/180000;    // Time step size
   int num_steps = static_cast<int>((t_final-t) / dt);

   ConstantCoefficient dbcCoef(1.0);
   Array<int> dbc_marker(mesh->bdr_attributes.Max());
   dbc_marker=0;
   dbc_marker[input-1] = 1; 
        
   // Solve the system: M_V * dV/dt = -K_V * I
   Vector rhs_V(Vnbrdof);
   Vector rhs_I(Inbrdof);
        
   // Temporary vectors
   Vector temp_V(Vnbrdof);
   Vector temp_I(Inbrdof);

   // Time-stepping loop (Euler Forward method)
   for (int step = 0; step < num_steps; step++)
   {
      // Update the voltage source at each time step
      VGF.ProjectBdrCoefficient(dbcCoef, dbc_marker);

      // Compute the right-hand side for V (Voltage update)
      M_V->Mult(VGF, rhs_V);  // Compute mass matrix * V
      K_IV->Mult(IGF, temp_V); // Compute K_V * I
      rhs_V += temp_V;     // Subtract K_V * I

      // Compute the right-hand side for I (Current update)
      M_I->Mult(IGF, rhs_I);  // Compute mass matrix * I
      K_VI->Mult(VGF, temp_I); // Compute K_I * V
      rhs_I += temp_I;     // Subtract K_I * V

      IGF.Add(dt, rhs_I);
      VGF.Add(dt, rhs_V);
  
      // Increment time
      t += dt;
}

 // Output the solution at each step
   Glvis(mesh, &VGF, "voltage", 8, "keys 'jRR'" );
   Glvis(mesh, &IGF, "current", 8, "keys 'jRR'" );
         

   return 1;

}


int TransmissionLineTransient::CreaterhsVector()
{
// computing rhs 
/*
   rhs = new Vector(nbrdof);
   *rhs = 0.0;

   std::ofstream out("out/rhs.txt");
   if(printMatrix) rhs->Print(out, 10);

   cout << rhs->Size() << " rhs size\n\n ";
*/
   return 1;

}

int TransmissionLineTransient::CreatexVector()
{
   /*
   x = new Vector(nbrdof);
   *x = 0.0; 
   std::ofstream out("out/x.txt");
   if(printMatrix) x->Print(out, 10);
   cout << x->Size() << " x size\n\n ";
   */
   return 1;
}
   
int TransmissionLineTransient::CreatePreconditionner()
{

   return 1;
}

int TransmissionLineTransient::Solver()
{
   // Solve system Ax = b
/*
   {
      std::ofstream out("out/xsol.txt");
      if(printMatrix) x->Print(out, 10);
   }
*/
   return 1;
}


int TransmissionLineTransient::PostPrecessing()
{
   return 1;
}

int TransmissionLineTransient::DisplayResults()
{

   return 1;
}

int TransmissionLineTransient::CleanOutDir()
{
    system("rm -f out/*");
    return 1;
}

int TransmissionLineTransient::Save()
{
/*
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
   */
   return 1;
}


int main(int argc, char *argv[])
{

   TransmissionLineTransient TLT;

   TLT.Parser(argc, argv);
   TLT.CleanOutDir();
   TLT.LoadMeshFile();
   TLT.CreateFESpace();
   TLT.CreateEssentialBoundary();
   TLT.CreateBilinears();
   TLT.TimeSteps();
   TLT.CreaterhsVector();
   TLT.CreatexVector();
   TLT.CreatePreconditionner();
   TLT.Solver();
   TLT.PostPrecessing();
   TLT.DisplayResults();
   TLT.Save();
   return 0;
}
