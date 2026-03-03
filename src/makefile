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

all: stltferk4 stltfdrk4 stltfe1d_04 stltfe1d_05

stltferk4: stltferk4.cpp MyTools.cpp
	g++ -g -O0 -o stltferk4  -std=c++11 -I$(MFEM_DIR) stltferk4.cpp MyTools.cpp -L$(MFEM_BUILD_DIR) \
  -lmfem -I/usr/local/include -I/usr/local/gmsh/include -L/usr/local/gmsh/lib -lgmsh -L/usr/lib -lgmsh -lrt 

stltfdrk4: stltfdrk4.cpp MyTools.cpp
	g++ -g -o stltfdrk4  -std=c++11 -I$(MFEM_DIR) stltfdrk4.cpp MyTools.cpp -L$(MFEM_BUILD_DIR) -lmfem -I/usr/local/include -lrt 

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
  -o stltfe1d_04

stltfe1d_05: stltfe1d_05.cpp MyTools.cpp
	g++ -g -O0 -std=c++11 \
  -I/mnt/c/mfem/mfem-4.8 \
   -I/mnt/c/mfem/mfem-4.8/../SuiteSparse/include \
  -I$(MFEM_DIR48)/../hypre-2.33.0/src/hypre/include \
  stltfe1d_05.cpp MyTools.cpp \
  -L/mnt/c/mfem/mfem-4.8 -lmfem \
  -L/usr/lib/x86_64-linux-gnu \
  -lumfpack -lcholmod -lcamd -lccolamd -lcolamd -lamd -lklu -lsuitesparseconfig \
  -L$(MFEM_DIR48)/../hypre-2.33.0/src/hypre/lib -lHYPRE -L$(MFEM_DIR48)/../metis-4.0.3 -lmetis \
  -lblas -llapack -lmetis \
  -o stltfe1d_05

