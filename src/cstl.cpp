//                                cstl
//                                
// compile with: make cstl, need gmsh.
//
// Sample runs:  ./cstl
//
/*
   This program generate mesh file representing a single transmission line.

*/

#include <iostream>
#include <math.h>
#include <filesystem>
#include "gmsh.h"


int main(int argc, char *argv[])
{
   // Before using any functions in the C++ API, gmsh::must be initialized:
   gmsh::initialize();

   gmsh::model::add("tlmesh");

   int point[10];
   int line[10];

   point[0] = gmsh::model::occ::addPoint(0, 0, 0);
   point[1] = gmsh::model::occ::addPoint(1, 0, 0);
   
   
   line[0] = gmsh::model::occ::addLine(point[0], point[1]);
   
    
   gmsh::model::occ::synchronize();

   gmsh::model::addPhysicalGroup(1, {line[0]}, 1, "tl1");
   gmsh::model::addPhysicalGroup(0, {point[0]}, 2, "input");
   gmsh::model::addPhysicalGroup(0, {point[1]}, 3, "output");
         
   gmsh::model::mesh::setOrder(1);

   // glvis can read mesh version 2.2
   gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);

   gmsh::model::mesh::generate(1);
   gmsh::model::mesh::refine();
   gmsh::model::mesh::refine();
   gmsh::model::mesh::refine();
   gmsh::model::mesh::refine();
   gmsh::model::mesh::refine();
   gmsh::model::mesh::refine();
   

   // ... and save it to disk
   gmsh::write("stlmesh-1.msh");

   // start gmsh
   gmsh::fltk::run();

   //before leaving.
   gmsh::finalize();
   
   return 0;
}
