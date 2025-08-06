
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


