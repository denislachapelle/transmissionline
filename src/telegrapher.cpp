#include <mfem.hpp>
#include <cmath>
#include <iostream>

using namespace mfem;
using namespace std;

static double GaussianPulse(double t, double amp, double fwhm, double t0)
{
   const double sigma = fwhm / (2.0 * std::sqrt(2.0 * std::log(2.0)));
   const double a = (t - t0) / sigma;
   return amp * std::exp(-(a*a));
}

int main(int argc, char *argv[])
{
   // Defaults
   int order = 1;
   int nx = 200, ny = 200;
   double length = 100.0;
   double t_final = 1e-6;

   // RG-58-ish parameters
   double R = 0.0483;     // ohm/m
   double L = 2.54e-7;    // H/m
   double G = 0.0;        // S/m
   double C = 1.01e-10;   // F/m

   // Source V(0,t): Gaussian pulse in time
   double V_amp = 1.0;
   double fwhm  = 30e-9;
   double t0 = -1.0; // if <0 pick 5*sigma

   const char vishost[] = "localhost";
   int visport = 19916;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "H1 polynomial order.");
   args.AddOption(&nx, "-nx", "--elements-x", "Elements along x (cable).");
   args.AddOption(&ny, "-ny", "--elements-t", "Elements along y (time).");
   args.AddOption(&length, "-l", "--length", "Cable length [m].");
   args.AddOption(&t_final, "-tf", "--t-final", "Final time [s].");
   args.AddOption(&R, "-R", "--resistance", "R [ohm/m].");
   args.AddOption(&L, "-L", "--inductance", "L [H/m].");
   args.AddOption(&G, "-G", "--conductance", "G [S/m].");
   args.AddOption(&C, "-C", "--capacitance", "C [F/m].");
   args.AddOption(&V_amp, "-Va", "--source-amplitude", "Source amplitude [V].");
   args.AddOption(&fwhm, "-fwhm", "--fwhm", "Gaussian FWHM [s].");
   args.AddOption(&t0, "-t0", "--pulse-center", "Pulse center [s] (<0 => auto).");
   args.AddOption(&visport, "-vp", "--vis-port", "GLVis server port.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);

   const double sigma = fwhm / (2.0 * std::sqrt(2.0 * std::log(2.0)));
   if (t0 < 0.0) { t0 = 5.0*sigma; }

   // 1) 2D mesh: x in [0,L], y in [0,T] where y is time
   // Boundary attrs: 1:left(x=0), 2:right(x=L), 3:bottom(y=0), 4:top(y=T)
   Mesh mesh = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL,
                                     true, length, t_final);

   // 2) Scalar H1 space (used for both I and V)
   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);
   const int n = fes.GetVSize();
   cout << "Number of DOFs per field: " << n << endl;

   // Coefficients
   ConstantCoefficient Lc(L), Rc(R), Cc(C), Gc(G);
   ConstantCoefficient one(1.0);
   // Small epsilon mass to avoid fully zero diagonal in pure derivative blocks
   ConstantCoefficient eps(1e-12);

   // 3) Assemble blocks
   // Eq1: L * dI/dy + R*I + dV/dx = 0
   BilinearForm a11(&fes);
   a11.AddDomainIntegrator(new DerivativeIntegrator(Lc, /*dir y=*/1));
   a11.AddDomainIntegrator(new MassIntegrator(Rc));
   a11.AddDomainIntegrator(new MassIntegrator(eps)); // numeric safety
   a11.Assemble(); a11.Finalize();
   SparseMatrix &A11 = a11.SpMat();

   // Coupling operator Dx: d()/dx
   BilinearForm dx(&fes);
   dx.AddDomainIntegrator(new DerivativeIntegrator(one, /*dir x=*/0));
   dx.Assemble(); dx.Finalize();
   SparseMatrix &Dx = dx.SpMat();

   // Eq2: C * dV/dy + G*V + dI/dx = 0
   BilinearForm a22(&fes);
   a22.AddDomainIntegrator(new DerivativeIntegrator(Cc, /*dir y=*/1));
   a22.AddDomainIntegrator(new MassIntegrator(Gc));
   a22.AddDomainIntegrator(new MassIntegrator(eps)); // numeric safety
   a22.Assemble(); a22.Finalize();
   SparseMatrix &A22 = a22.SpMat();

   // A12 = Dx (V -> Eq1 test)
   // A21 = Dx (I -> Eq2 test)

   // 4) Build monolithic A = [[A11, A12],[A21, A22]] using LIL SparseMatrix
   const int N = 2*n;
   SparseMatrix A(N, N); // LIL flexible; new entries are added as needed :contentReference[oaicite:6]{index=6}

   Array<int> cols;
   Vector srow;

   // Top block rows: Eq1
   for (int i = 0; i < n; i++)
   {
      A11.GetRow(i, cols, srow);
      for (int k = 0; k < cols.Size(); k++)
      {
         A.Add(i, cols[k], srow[k]);
      }

      Dx.GetRow(i, cols, srow);
      for (int k = 0; k < cols.Size(); k++)
      {
         A.Add(i, n + cols[k], srow[k]); // A12
      }
   }

   // Bottom block rows: Eq2
   for (int i = 0; i < n; i++)
   {
      const int row = n + i;

      Dx.GetRow(i, cols, srow);
      for (int k = 0; k < cols.Size(); k++)
      {
         A.Add(row, cols[k], srow[k]); // A21
      }

      A22.GetRow(i, cols, srow);
      for (int k = 0; k < cols.Size(); k++)
      {
         A.Add(row, n + cols[k], srow[k]);
      }
   }
// After filling A with A11, Dx, A21, A22 but BEFORE Finalize():
for (int i = 0; i < N; i++)
{
   A.Add(i, i, 0.0); // create diagonal entry if missing (value stays unchanged)
}
   A.Finalize(); // convert to CSR :contentReference[oaicite:7]{index=7}

   // 5) Build RHS (homogeneous PDE), and impose essential BCs by elimination
   Vector rhs(N); rhs = 0.0;

   // Essential boundary markers for I and V
   const int nbdr = mesh.bdr_attributes.Max();
   Array<int> ess_bdr_I(nbdr), ess_bdr_V(nbdr);
   ess_bdr_I = 0; ess_bdr_V = 0;

   // I(x,0)=0 on bottom (attr 3)
   if (nbdr >= 3) { ess_bdr_I[2] = 1; }

   // V(x,0)=0 on bottom (attr 3), V(0,t)=pulse on left (attr 1), V(L,t)=0 on right (attr 2)
   if (nbdr >= 3) { ess_bdr_V[2] = 1; }
   if (nbdr >= 1) { ess_bdr_V[0] = 1; }
   if (nbdr >= 2) { ess_bdr_V[1] = 1; }

   Array<int> ess_vdofs_I, ess_vdofs_V;
   fes.GetEssentialVDofs(ess_bdr_I, ess_vdofs_I);
   fes.GetEssentialVDofs(ess_bdr_V, ess_vdofs_V);

   // Prescribed V boundary values
   FunctionCoefficient Vsrc([&](const Vector &p)
   {
      const double t = p(1); // y is time
      return GaussianPulse(t, V_amp, fwhm, t0);
   });

   GridFunction Vbc(&fes);
   Vbc = 0.0;
   Vbc.ProjectBdrCoefficient(Vsrc, ess_bdr_V);

   // Apply elimination: for each essential DOF, eliminate row+col and update rhs using sol
   // This moves BC effect to RHS and keeps matrix size. :contentReference[oaicite:8]{index=8}
   for (int i = 0; i < n; i++)
   {
      if (ess_vdofs_I[i])
      {
         const int rc = i;          // I block
         const double sol = 0.0;    // I=0
         //A.EliminateRowCol(rc, sol, rhs, DIAG_ONE);
         A.EliminateRowCol(rc, sol, rhs, mfem::Operator::DIAG_ONE);

      }
   }
   for (int i = 0; i < n; i++)
   {
      if (ess_vdofs_V[i])
      {
         const int rc = n + i;      // V block
         const double sol = Vbc(i); // V prescribed
         A.EliminateRowCol(rc, sol, rhs, mfem::Operator::DIAG_ONE);
         cout << i << endl;
      }
   }

   // 6) Solve A x = rhs
   Vector x(N); x = 0.0;

   GMRESSolver solver;
   solver.SetOperator(A);
   solver.SetRelTol(1e-8);
   solver.SetAbsTol(0.0);
   solver.SetMaxIter(5000);
   solver.SetPrintLevel(1);

   // No diagonal-based preconditioner here (your previous runs showed diagonal issues).
   solver.Mult(rhs, x);

   // 7) Split into I and V GridFunctions
   GridFunction I_sol(&fes), V_sol(&fes);
   for (int i = 0; i < n; i++)
   {
      I_sol(i) = x(i);
      V_sol(i) = x(n + i);
   }

   // 8) Send I and V separately to GLVis server
   {
      socketstream sock(vishost, visport);
      if (sock)
      {
         sock.precision(16);
         sock << "solution\n" << mesh << I_sol << flush;
      }
      else { cout << "Could not connect to GLVis for I.\n"; }
   }
   {
      socketstream sock(vishost, visport);
      if (sock)
      {
         sock.precision(16);
         sock << "solution\n" << mesh << V_sol << flush;
      }
      else { cout << "Could not connect to GLVis for V.\n"; }
   }

   return 0;
}
