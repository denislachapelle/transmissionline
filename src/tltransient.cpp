//                                tltransient
//                                
// Compile with: make tltransient, need MFEM version 4.7 and GLVIS-4.3.
//
// Sample runs:  ./tltransient
//
/*
Description:  

*/

/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = -v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
    class TLOperator : public TimeDependentOperator
    {
    private:
       BilinearForm &M, &K;
       const Vector &b;
       Solver *M_prec;
       CGSolver M_solver;
       DG_Solver *dg_solver;
    
       mutable Vector z;
    
    public:
       TLOperator(BilinearForm &M_, BilinearForm &K_, const Vector &b_);
    
       void Mult(const Vector &x, Vector &y) const override;
       void ImplicitSolve(const real_t dt, const Vector &x, Vector &k) override;
    
       ~TLOperator() override;
    };

    // Implementation of class FE_Evolution
    TLOperator::FE_Evolution(BilinearForm &M_, BilinearForm &K_, const Vector &b_)
: TimeDependentOperator(M_.FESpace()->GetTrueVSize()),
  M(M_), K(K_), b(b_), z(height)
{
Array<int> ess_tdof_list;
if (M.GetAssemblyLevel() == AssemblyLevel::LEGACY)
{
   M_prec = new DSmoother(M.SpMat());
   M_solver.SetOperator(M.SpMat());
   dg_solver = new DG_Solver(M.SpMat(), K.SpMat(), *M.FESpace());
}
else
{
   M_prec = new OperatorJacobiSmoother(M, ess_tdof_list);
   M_solver.SetOperator(M);
   dg_solver = NULL;
}
M_solver.SetPreconditioner(*M_prec);
M_solver.iterative_mode = false;
M_solver.SetRelTol(1e-9);
M_solver.SetAbsTol(0.0);
M_solver.SetMaxIter(100);
M_solver.SetPrintLevel(0);
}

void TLOperator::Mult(const Vector &x, Vector &y) const
{
// y = M^{-1} (K x + b)
K.Mult(x, z);
z += b;
M_solver.Mult(z, y);
}

void TLOperator::ImplicitSolve(const real_t dt, const Vector &x, Vector &k)
{
MFEM_VERIFY(dg_solver != NULL,
            "Implicit time integration is not supported with partial assembly");
K.Mult(x, z);
z += b;
dg_solver->SetTimeStep(dt);
dg_solver->Mult(z, k);
}

TLOperator::~FE_Evolution()
{
delete M_prec;
delete dg_solver;
}

    