//                                TwoDWiresGenerator
//   

#ifndef TWODWIRESGENERATOR
#define TWODWIRESGENERATOR

#include "mytools.hpp"
#include <iostream>
#include <math.h>
#include <filesystem>
#include "gmsh.h"

using namespace std;

class GroupInfo
{
   public:
   double current[2];
   int nbrWires;
};

class WireInfo
{
   public:
   int group;
   enum WireTypes {roundwire, rectangular, awg};
   WireTypes type;
   double dimensions[10];
   double center[2];
   double current[2];
};

class TwoDWiresGenerator
{

   protected:
      

   private:
      // 1. Parse command-line options.
      const char *configFile = "fourwires.txt";
      const char *meshFile = "fourwires.msh";
      int nbrwires = 2;
      WireInfo *wiresInfo;
      int nbrGroups;
      GroupInfo *groupsInfo;

      real_t domainRadius = -1.0;
      int refineTimes = 0;
      int meshElementOrder = 1;

   public:
      //parse the options.
      int Parser(int argc, char *argv[]);

      int ReadConfigFile();
      int CreateMeshFile();
      int GetNbrGroups();
      GroupInfo *GetGroupsInfo();

};
#endif //2DWIRESGENERATOR
