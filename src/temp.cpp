/*
class BlockDiagAMG : public Solver
{
public:
   BlockDiagAMG(const Array<int> &offsets,
                HypreParMatrix &A00,
                HypreParMatrix &A11)
      : Solver(offsets.Last()),
        offsets_(offsets),
        amg0_(&A00), amg1_(&A11),
        x0_(offsets[1]-offsets[0]),
        x1_(offsets[2]-offsets[1]),
        y0_(offsets[1]-offsets[0]),
        y1_(offsets[2]-offsets[1])
   {
      // Configure each BoomerAMG as you like
      amg0_.SetPrintLevel(0);
      amg0_.SetSystemsOptions(1);  // if A00 is vector-valued SPD; else omit
      amg0_.SetCoarsenType(8);     // HMIS coarsening (example)
      amg0_.SetRelaxType(6);       // Symmetric SOR/Jacobi (example)

      amg1_.SetPrintLevel(0);
      amg1_.SetCoarsenType(8);
      amg1_.SetRelaxType(6);

      // Recommended: amg0_.SetOperator(A00); amg1_.SetOperator(A11);  // done by constructors

      // This preconditioner is square and maps size N -> N
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      // Split input into blocks
      Vector x0(const_cast<double*>(x.GetData()+offsets_[0]), offsets_[1]-offsets_[0]);
      Vector x1(const_cast<double*>(x.GetData()+offsets_[1]), offsets_[2]-offsets_[1]);

      // Apply AMG per block
      amg0_.Mult(x0, y0_);
      amg1_.Mult(x1, y1_);

      // Stitch output
      y.SetSize(offsets_.Last());
      y.Range(offsets_[0], offsets_[1]-offsets_[0]) = y0_;
      y.Range(offsets_[1], offsets_[2]-offsets_[1]) = y1_;
   }

   // Optional: expose SetOperator to rebuild AMG after updates
   void SetOperators(HypreParMatrix &A00, HypreParMatrix &A11)
   {
      amg0_.SetOperator(A00);
      amg1_.SetOperator(A11);
   }

private:
   const Array<int> &offsets_;
   mutable HypreBoomerAMG amg0_, amg1_;
   mutable Vector x0_, x1_, y0_, y1_;
};
*/


      {
      std::ofstream out("out/b1.txt");
      if(printMatrix) b.Print(out, 10);
      }


      {
      std::ofstream out("out/b2.txt");
      if(printMatrix) b.Print(out, 10);
      }

      if(Time >= plotCount * plotTime)
      {
         std::string s = "out/x" + std::to_string(plotCount) + ".txt";
         std::ofstream out(s);
         Vector val(xBlock->GetBlock(0));
         val.Print(out, 1);
         plotCount++;
         cout << Time << " Time" << endl;
         cout << val.Max() << " Max" << endl;
      }

      //Time += deltaT;

      
   
   {
      std::ostringstream oss;
       oss << "octave --persist --quiet --eval \"figure(1); hold on; ";
    
      for(int i=1; i<nbrPlot; i++) 
      {
         oss << "subplot(5,1, " << i << "); " << "plot(x" << i << "=load('out/x" << i << ".txt')); ";
      }
      oss << "input('Press Enter to close'); \"";
      std::string result = oss.str();
      system(result.c_str());
      
   //system("gnuplot -persist -e \"plot 'out/x1.txt' with lines title 'x1', \
                                      'out/x2.txt' with lines title 'x2', \
                                      'out/x3.txt' with lines title 'x3', \
                                      'out/x4.txt' with lines title 'x4'\"");
   }


