//                                TwoDWiresGenerator
//                                
// compile with: make TwoDWiresGenerator, need gmsh.
//
// Sample runs:  ./TwoDWiresGenerator -cf wiresconfig.txt
//
/*
Description:  

*/



#include "mytools.hpp"
#include <iostream>
#include <math.h>
#include <filesystem>
#include "gmsh.h"
#include "twodwiresgenerator.hpp"

using namespace std;

 int TwoDWiresGenerator::GetNbrGroups() {return nbrGroups;}
 GroupInfo *TwoDWiresGenerator::GetGroupsInfo() {return groupsInfo;}


int TwoDWiresGenerator::Parser(int argc, char *argv[])
{

   OptionsParser args(argc, argv);
   args.AddOption(&meshFile, "-mf", "--meshfile",
                  "file to use as mesh file.");
   args.AddOption(&configFile, "-cf", "--configfile",
                  "file to use as wires config file.");
   args.AddOption(&refineTimes, "-rft", "--refinetimes",
                  "Number of times gmsh refine.");
   args.AddOption(&meshElementOrder, "-meo", "--meshelementorder",
                  "Order of the mesh elements.");
                                 
                
   args.Parse();

   if (args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   return 0;
}


int TwoDWiresGenerator::ReadConfigFile()
{
   std::string s;
   std::stringstream ss;
   nbrwires = HowManyWire(configFile);
   cout << nbrwires << " nbrWire\n";
   assert(nbrwires>0);

   if(nbrwires>0) wiresInfo = new WireInfo[nbrwires];
 
   for(int wc=0; wc<nbrwires; wc++)
   {
      ss.str("");
      ss << "wire" << wc+1 << "center";
      s = findWordN(configFile, ss.str(), 2);
      wiresInfo[wc].center[0] = stof(s);

      ss.str("");
      ss << "wire" << wc+1 << "center";
      s = findWordN(configFile, ss.str(), 3);
      wiresInfo[wc].center[1] = stof(s);

      ss.str("");
      ss << "wire" << wc+1 << "group";
      s = findWordN(configFile, ss.str(), 2);
      wiresInfo[wc].group = stof(s);
         
      ss.str("");
      ss << "wire" << wc+1 << "shape";
      s = findWordN(configFile, ss.str(), 2);
      if(s =="round")
      {
         wiresInfo[wc].type = WireInfo::WireTypes::roundwire;

         ss.str("");
         ss << "wire" << wc+1 << "dimensions";
         s = findWordN(configFile, ss.str(), 2);
         wiresInfo[wc].dimensions[0] = stof(s);

      }
      else if(s =="rectangular")
      {
         wiresInfo[wc].type = WireInfo::WireTypes::rectangular;

         ss.str("");
         ss << "wire" << wc+1 << "dimensions";
         s = findWordN(configFile, ss.str(), 2);
         wiresInfo[wc].dimensions[0] = stof(s);
         
         ss.str("");
         ss << "wire" << wc+1 << "dimensions";
         s = findWordN(configFile, ss.str(), 3);
         wiresInfo[wc].dimensions[1] = stof(s);
      }
      else if(s =="awg")
      {
         wiresInfo[wc].type = WireInfo::WireTypes::awg;

         ss.str("");
         ss << "wire" << wc+1 << "dimensions";
         s = findWordN(configFile, ss.str(), 2);
         double dim = stof(s);
         double diameter = 0.000127 * pow(92, (36-dim)/39);
         wiresInfo[wc].dimensions[0] = diameter/2;
      }
   }

   int groupMax=0;
   for(int wc=0; wc<nbrwires; wc++)
   {
      if(wiresInfo[wc].group>groupMax) groupMax=wiresInfo[wc].group;
   }
   nbrGroups = groupMax;

   groupsInfo = new GroupInfo[nbrGroups];
   
   for(int gc=0; gc<nbrGroups; gc++)
   {
      ss.str("");
      ss << "group" << gc+1 << "current";
      s = findWordN(configFile, ss.str(), 2);
      groupsInfo[gc].current[0] = stof(s);

      ss.str("");
      ss << "group" << gc+1 << "current";
      s = findWordN(configFile, ss.str(), 3);
      groupsInfo[gc].current[1] = stof(s);
   }
   
   ss.str("");
   ss << "domainradius";
   s = findWordN(configFile, ss.str(), 2);
   domainRadius = stof(s);
   
   return 1;
}


int TwoDWiresGenerator::CreateMeshFile()
{
   // Before using any functions in the C++ API, gmsh::must be initialized:
   gmsh::initialize();

   gmsh::model::add("proxmesh");
   int wc;
   int *wtag;
   wtag = new int[nbrwires];

   for(wc=0; wc<nbrwires; wc++)
   {
      if(wiresInfo[wc].type==WireInfo::WireTypes::roundwire || wiresInfo[wc].type==WireInfo::WireTypes::awg)
      {
         int circleTag =gmsh::model::occ::addCircle(wiresInfo[wc].center[0], wiresInfo[wc].center[1], 0, wiresInfo[wc].dimensions[0], -1);
         int curveLoopTag = gmsh::model::occ::addCurveLoop({circleTag}, -1);
         wtag[wc] = gmsh::model::occ::addPlaneSurface({curveLoopTag}, -1);
      }
      else if(wiresInfo[wc].type==WireInfo::WireTypes::rectangular)
      {
         wtag[wc] = gmsh::model::occ::addRectangle(wiresInfo[wc].center[0] - wiresInfo[wc].dimensions[0]/2.0,
                                              wiresInfo[wc].center[1] - wiresInfo[wc].dimensions[1]/2.0,
                                              0,
                                              wiresInfo[wc].dimensions[0],
                                              wiresInfo[wc].dimensions[1],
                                              -1,
                                              0);
      }

   }
   gmsh::model::occ::synchronize();

   //domain limit

   int circleTag = gmsh::model::occ::addCircle(0, 0, 0, domainRadius, -1);

   int curveLoopTag = gmsh::model::occ::addCurveLoop({circleTag}, -1);

   std::vector<int> vec(nbrwires+1);
   vec.at(0)=curveLoopTag;
   for(wc=0; wc<nbrwires; wc++ )
   {
      vec.at(wc+1)=-(wtag[wc]);
   }
   
   int domainTag = gmsh::model::occ::addPlaneSurface(vec, -1);

   //
   //synchronize prior to add physical group.
   //
   gmsh::model::occ::synchronize();
   //
   //add physical groups.
   //
  
   for(int gc=0; gc<nbrGroups; gc++)
   {
      char s[10];
      sprintf(s, "group_%d",  gc+1);
      std::vector<int> groupTags;
      for(wc=0; wc<nbrwires; wc++)
      {
         if(wiresInfo[wc].group==gc+1) groupTags.push_back(wtag[wc]);
      }
      gmsh::model::addPhysicalGroup(2, groupTags, gc+1, s);
   }

   gmsh::model::addPhysicalGroup(2, {domainTag}, nbrGroups+1 , "air");
   gmsh::model::addPhysicalGroup(1, {circleTag}, nbrGroups+2, "aircontour");

   // We can then generate a 2D mesh...
 //  gmsh::option::setNumber("Mesh.Algorithm", 6);


   gmsh::model::mesh::generate(2);
   for(int i = 0; i < refineTimes; i++)
   {
      gmsh::model::mesh::refine();
   }
      
   gmsh::model::mesh::setOrder(meshElementOrder);

   // glvis can read mesh version 2.2
   gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);

   // ... and save it to disk
   gmsh::write(meshFile);

   // start gmsh
   gmsh::fltk::run();

   //before leaving.
   gmsh::finalize();

   return 1;
}
