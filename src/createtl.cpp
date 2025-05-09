//                                createwires2dmesh
//                                
// compile with: make createtl, need gmsh.
//
// Sample runs:  ./createtl
//
/*
   This program generate mesh file representing a transmission line.

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
   point[1] = gmsh::model::occ::addPoint(29, 0, 0);
   point[2] = gmsh::model::occ::addPoint(30.0, 0, 0);
   point[3] = gmsh::model::occ::addPoint(29, 29.1, 0);
   
   line[0] = gmsh::model::occ::addLine(point[0], point[1]);
   line[1] = gmsh::model::occ::addLine(point[1], point[2]);
   line[2] = gmsh::model::occ::addLine(point[1], point[3]);
   
    
   gmsh::model::occ::synchronize();

   gmsh::model::addPhysicalGroup(1, {line[0], line[1]}, 1, "tl1");
   gmsh::model::addPhysicalGroup(1, {line[2]}, 2, "tl2");
   gmsh::model::addPhysicalGroup(0, {point[0]}, 3, "input");
   gmsh::model::addPhysicalGroup(0, {point[2]}, 4, "output");
   gmsh::model::addPhysicalGroup(0, {point[3]}, 5, "stubend");
         
   gmsh::model::mesh::setOrder(1);

   // glvis can read mesh version 2.2
   gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);

   gmsh::model::mesh::generate(1);
   gmsh::model::mesh::refine();
   gmsh::model::mesh::refine();

   // ... and save it to disk
   gmsh::write("tlmesh.msh");

   // start gmsh
   gmsh::fltk::run();

   //before leaving.
   gmsh::finalize();
   
   return 0;
}
