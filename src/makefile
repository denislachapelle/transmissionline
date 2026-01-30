#
# DL241118
# this file compile all programs developped while experimenting with mfem.
#

MFEM_DIR ?= /mnt/c/mfem-4.7
MFEM_DIR48 ?= /mnt/c/mfem/mfem-4.8
MFEM_BUILD_DIR ?= /mnt/c/mfem-4.7
MFEM_BUILD_DIR48 ?= /mnt/c/mfem/mfem-4.8

GMSH_DIR ?= /
#MFEM_DIR ?= /home/denislachapelle2003/fem/mfem-4.6
#MFEM_BUILD_DIR ?= /home/denislachapelle2003/fem/mfem-4.6

#COMMON_LIB = -L$(MFEM_BUILD_DIR)/miniapps/common -lmfem-common

all: createtl tltransient tlchatgpt stlt sltfe stltferk4 stltfdrk4 stltfe2d_submesh stltfe2d_p stltfe2d_NO_submesh stltfe2d_submesh_p stltfe1d_04

createtl: createtl.cpp 
	g++ -g -o ../createtl  -std=c++11 -I$(MFEM_DIR) createtl.cpp -L$(MFEM_BUILD_DIR) -lmfem -I/usr/local/include -L/usr/lib/libgmsh.so -lgmsh -lrt 

tltransient: tltransient.cpp MyTools.cpp
	g++ -g -o ../tltransient  -std=c++11 -I$(MFEM_DIR) tltransient.cpp MyTools.cpp -L$(MFEM_BUILD_DIR) -lmfem -I/usr/local/include -L/usr/lib/libgmsh.so -lgmsh -lrt 

tlchatgpt: tlchatgpt.cpp 
	g++ -g -o ../tlchatgpt  -std=c++11 -I$(MFEM_DIR) tlchatgpt.cpp -L$(MFEM_BUILD_DIR) -lmfem -I/usr/local/include -L/usr/lib/libgmsh.so -lgmsh -lrt 

stlt: stlt.cpp MyTools.cpp
	g++ -g -o ../stlt  -std=c++11 -I$(MFEM_DIR) stlt.cpp MyTools.cpp -L$(MFEM_BUILD_DIR) -lmfem -I/usr/local/include -L/usr/lib/libgmsh.so -lgmsh -lrt 

stltfe: stltfe.cpp MyTools.cpp
	g++ -g -o ../stltfe  -std=c++11 -I$(MFEM_DIR) stltfe.cpp MyTools.cpp -L$(MFEM_BUILD_DIR) -lmfem -I/usr/local/include -L/usr/lib/libgmsh.so -lgmsh -lrt 

stltferk4: stltferk4.cpp MyTools.cpp
	g++ -g -O0 -o ../stltferk4  -std=c++11 -I$(MFEM_DIR) stltferk4.cpp MyTools.cpp -L$(MFEM_BUILD_DIR) -lmfem -I/usr/local/include -I/usr/local/gmsh/include -L/usr/local/gmsh/lib -lgmsh -L/usr/lib -lgmsh -lrt 

stltfdrk4: stltfdrk4.cpp MyTools.cpp
	g++ -g -o ../stltfdrk4  -std=c++11 -I$(MFEM_DIR) stltfdrk4.cpp MyTools.cpp -L$(MFEM_BUILD_DIR) -lmfem -I/usr/local/include -lrt 

stltfd: stltfd.cpp MyTools.cpp
	g++ -g -o ../stltfd  -std=c++11 -I$(MFEM_DIR) stltfd.cpp MyTools.cpp -L$(MFEM_BUILD_DIR) -lmfem -I/usr/local/include -L/usr/lib/libgmsh.so -lgmsh -lrt 


cstl: cstl.cpp 
	g++ -g -o ../cstl  -std=c++11 -I$(MFEM_DIR) cstl.cpp -L$(MFEM_BUILD_DIR) -lmfem -I/usr/local/include -L/usr/lib/libgmsh.so -lgmsh -lrt 


stltfe2d_p: stltfe2d_p.cpp MyTools.cpp
	mpicxx  -g -O0 -Wall -std=c++11 -I$(MFEM_DIR48) -I$(MFEM_DIR48)/../hypre/src/hypre/include stltfe2d_p.cpp MyTools.cpp -o ../stltfe2d_p -L$(MFEM_BUILD_DIR48) -lmfem -L$(MFEM_DIR48)/../hypre-2.33.0/src/hypre/lib -lHYPRE -L$(MFEM_DIR48)/../metis-4.0.3 -lmetis -lrt

#stltfe2d_submesh: stltfe2d_submesh.cpp MyTools.cpp
#	g++ -g -o0 -o ../stltfe2d_submesh  -std=c++11 -I$(MFEM_DIR48) stltfe2d_submesh.cpp MyTools.cpp -L$(MFEM_BUILD_DIR48) -lmfem -I/usr/local/include -I/usr/local/gmsh/include -I$(MFEM_DIR48)/../SuiteSparse/include -L/usr/lib/x86_64-linux-gnu -L/usr/local/gmsh/lib -lgmsh -L/usr/lib -lgmsh -lrt 

stltfe2d_submesh: stltfe2d_submesh.cpp MyTools.cpp
	g++ -g -O0 -std=c++11 \
  -I/mnt/c/mfem/mfem-4.8 \
  -I/usr/local/include -I/usr/local/gmsh/include \
  -I/mnt/c/mfem/mfem-4.8/../SuiteSparse/include \
  -I$(MFEM_DIR48)/../hypre-2.33.0/src/hypre/include \
  stltfe2d_submesh.cpp MyTools.cpp \
  -L/mnt/c/mfem/mfem-4.8 -lmfem \
  -L/usr/local/lib -L/usr/local/gmsh/lib -lgmsh \
  -L/usr/lib/x86_64-linux-gnu \
  -lumfpack -lcholmod -lcamd -lccolamd -lcolamd -lamd -lklu -lsuitesparseconfig \
  -L$(MFEM_DIR48)/../hypre-2.33.0/src/hypre/lib -lHYPRE -L$(MFEM_DIR48)/../metis-4.0.3 -lmetis \
  -lblas -llapack -lmetis \
  -o ../stltfe2d_submesh

stltfe2d_NO_submesh: stltfe2d_NO_submesh.cpp MyTools.cpp
	g++ -g -o0 -o ../stltfe2d_NO_submesh  -std=c++11 -I$(MFEM_DIR48) stltfe2d_NO_submesh.cpp MyTools.cpp -L$(MFEM_BUILD_DIR48) -lmfem -I/usr/local/include -I/usr/local/gmsh/include -L/usr/local/gmsh/lib -lgmsh -L/usr/lib -lgmsh -lrt 




stltfe2d_submesh_p: stltfe2d_submesh_p.cpp MyTools.cpp
	mpicxx  -g -O0 -Wall -std=c++11 \
	-I$(MFEM_DIR48) \
	-I$(MFEM_DIR48)/../hypre/src/hypre/include \
	stltfe2d_submesh_p.cpp MyTools.cpp -o ../stltfe2d_submesh_p \
	-L$(MFEM_BUILD_DIR48) -lmfem \
	-L$(MFEM_DIR48)/../hypre-2.33.0/src/hypre/lib -lHYPRE \
	-L$(MFEM_DIR48)/../metis-4.0.3 -lmetis \
	-lrt

stltfe1d_04: stltfe1d_04.cpp MyTools.cpp
	g++ -g -O0 -std=c++11 \
  -I/mnt/c/mfem/mfem-4.8 \
   -I/mnt/c/mfem/mfem-4.8/../SuiteSparse/include \
  -I$(MFEM_DIR48)/../hypre-2.33.0/src/hypre/include \
  stltfe1d_04.cpp MyTools.cpp \
  -L/mnt/c/mfem/mfem-4.8 -lmfem \
  -L/usr/lib/x86_64-linux-gnu \
  -lumfpack -lcholmod -lcamd -lccolamd -lcolamd -lamd -lklu -lsuitesparseconfig \
  -L$(MFEM_DIR48)/../hypre-2.33.0/src/hypre/lib -lHYPRE -L$(MFEM_DIR48)/../metis-4.0.3 -lmetis \
  -lblas -llapack -lmetis \
  -o ../stltfe1d_04

