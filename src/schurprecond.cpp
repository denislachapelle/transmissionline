
/// Block-Jacobi preconditioner for the primal block A = BA.
///
/// A is a 2x2 BlockMatrix with blocks:
///   A00 : V-V coupling (MBLF_d
///   A11 : I-I coupling (MBLF_didx)
///
/// This Solver acts on the primal part only (size = A.Height()).
class PrimalBlockJacobi : public Solver
{
private:

   const BlockMatrix &A;
   mutable DSmoother invA00;
   mutable DSmoother invA11;
   int nV, nI;

public:
   PrimalBlockJacobi(BlockMatrix &A_)
      : Solver(A_.Height())
      , A(A_)
      , invA00(1, 1.0, 5)   // type=1 (Jacobi), 5 iterations (tune if needed)
      , invA11(1, 1.0, 5)
   {
      // Get sizes of V and I sub-blocks from A's row offsets.
      const Array<int> &roff = A.RowOffsets();
      MFEM_VERIFY(roff.Size() == 3, "Expected 2x2 BlockMatrix for A.");
      nV = roff[1] - roff[0];
      nI = roff[2] - roff[1];

      // Initialize smoothers on the diagonal blocks.
      invA00.SetOperator(A.GetBlock(0,0));
      invA11.SetOperator(A.GetBlock(1,1));
   }

   virtual void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_ASSERT(x.Size() == Size(), "PrimalBlockJacobi: size mismatch.");
      MFEM_ASSERT(y.Size() == Size(), "PrimalBlockJacobi: size mismatch.");

      const double *xd = x.Read();
      double *yd = y.Write();

      // Split x into [xV; xI], y into [yV; yI]
      Vector xV(const_cast<double *>(xd), nV);
      Vector xI(const_cast<double *>(xd) + nV, nI);

      Vector yV(yd, nV);
      Vector yI(yd + nV, nI);

      // Apply diagonal smoothers blockwise:
      invA00.Mult(xV, yV);
      invA11.Mult(xI, yI);
   }

   virtual void SetOperator(const Operator &op) override
   {
      MFEM_ABORT("PrimalBlockJacobi::SetOperator is not supported.");
   }
};

/// Schur complement operator: S = B * Ainv * B^T
/// - Ainv: Solver acting on primal space (size = A.Height()).
/// - B   : BlockOperator mapping primal → lambda (your BB).
///
/// This Operator maps lambda-space → lambda-space.
class SchurComplementOperator : public Operator
{
private:
   Solver &Ainv;           // approximate A^{-1}
   BlockOperator &B;       // constraint operator
   mutable Vector tmp_primal;
   mutable Vector tmp_lambda;

public:
   SchurComplementOperator(Solver &Ainv_, BlockOperator &B_)
      : Operator(B_.Height())
      , Ainv(Ainv_)
      , B(B_)
      , tmp_primal(B_.Width())
      , tmp_lambda(B_.Height())
   {
      MFEM_VERIFY(B_.Width()  == Ainv_.Height(),
                  "SchurComplementOperator: dimension mismatch between B and Ainv.");
   }

   /// y = S x = B * Ainv * B^T * x
   virtual void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_ASSERT(x.Size() == Height(), "SchurComplementOperator: input size mismatch.");
      MFEM_ASSERT(y.Size() == Height(), "SchurComplementOperator: output size mismatch.");

      // tmp_lambda = x  (lambda-space)
      tmp_lambda = x;

      // tmp_primal = B^T * x
      B.MultTranspose(tmp_lambda, tmp_primal);

      // tmp_primal = Ainv * tmp_primal
      Ainv.Mult(tmp_primal, tmp_primal);

      // y = B * tmp_primal
      B.Mult(tmp_primal, y);
   }
};

/// Full block Schur preconditioner for the 2x2 saddle-point system:
///
///   [ A  B^T ]
///   [ B   0  ]
///
/// Unknown ordering: [x_primal; x_lambda]
///
/// - A      : BlockMatrix (your BA)
/// - B      : BlockOperator (your BB)
/// - offsets: same Array<int> used in OuterBlock for block partitioning.
///            offsets[0]=0, offsets[1]=A.Height(), offsets[2]=A.Height()+B.Height().
class FullSchurPreconditioner : public Solver
{
private:
   Array<int> offsets;
   PrimalBlockJacobi *Aprec;
   SchurComplementOperator *SchurOp;
   GMRESSolver *SchurSolver;   // approximates S^{-1} on lambda-space

public:
   FullSchurPreconditioner(BlockMatrix &A, BlockOperator &B,
                           const Array<int> &outerOffsets)
      : Solver(outerOffsets[outerOffsets.Size()-1]) // total size
      , offsets(outerOffsets)
   {
      MFEM_VERIFY(offsets.Size() == 3,
                  "FullSchurPreconditioner: expected 2 blocks: primal, lambda");

      // 1) A-block preconditioner (primal)
      Aprec = new PrimalBlockJacobi(A);

      // 2) Schur complement operator S = B Aprec B^T
      SchurOp = new SchurComplementOperator(*Aprec, B);

      // 3) Solver for S^{-1} (lambda-space)
      SchurSolver = new GMRESSolver();
      SchurSolver->SetAbsTol(0.0);
      SchurSolver->SetRelTol(1e-1);    // loose tolerance OK, it's a preconditioner
      SchurSolver->SetMaxIter(50);
      SchurSolver->SetKDim(30);
      SchurSolver->SetPrintLevel(0);
      SchurSolver->SetOperator(*SchurOp);
   }

   virtual ~FullSchurPreconditioner()
   {
      delete SchurSolver;
      delete SchurOp;
      delete Aprec;
   }

   virtual void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_ASSERT(x.Size() == Size(), "FullSchurPreconditioner: input size mismatch.");
      MFEM_ASSERT(y.Size() == Size(), "FullSchurPreconditioner: output size mismatch.");

      const double *xd = x.Read();
      double *yd = y.Write();

      const int n_primal = offsets[1] - offsets[0];
      const int n_lambda = offsets[2] - offsets[1];

      // Split x into [xp; xl], y into [yp; yl]
      Vector xp(const_cast<double *>(xd), n_primal);
      Vector xl(const_cast<double *>(xd) + n_primal, n_lambda);

      Vector yp(yd, n_primal);
      Vector yl(yd + n_primal, n_lambda);

      // 1) yp = Â^{-1} xp
      Aprec->Mult(xp, yp);

      // 2) yl = Ŝ^{-1} xl   (via inner GMRES on SchurOp)
      SchurSolver->Mult(xl, yl);
   }

   virtual void SetOperator(const Operator &op) override
   {
      // This preconditioner is built for a fixed K, no dynamic update.
      MFEM_ABORT("FullSchurPreconditioner::SetOperator not implemented.");
   }
};
