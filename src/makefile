#
# DL241118
# this file compile all programs developped while experimenting with mfem.
#

MFEM_DIR ?= /mnt/c/mfem-4.7
MFEM_BUILD_DIR ?= /mnt/c/mfem-4.7

GMSH_DIR ?= /
#MFEM_DIR ?= /home/denislachapelle2003/fem/mfem-4.6
#MFEM_BUILD_DIR ?= /home/denislachapelle2003/fem/mfem-4.6

#COMMON_LIB = -L$(MFEM_BUILD_DIR)/miniapps/common -lmfem-common

all: createtl tltransient tlchatgpt stlt sltfe stltferk4 stltfdrk4

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
	g++ -g -o ../stltferk4  -std=c++11 -I$(MFEM_DIR) stltferk4.cpp MyTools.cpp -L$(MFEM_BUILD_DIR) -lmfem -I/usr/local/include -L/usr/lib/libgmsh.so -lgmsh -lrt 

stltfdrk4: stltfdrk4.cpp MyTools.cpp
	g++ -g -o ../stltfdrk4  -std=c++11 -I$(MFEM_DIR) stltfdrk4.cpp MyTools.cpp -L$(MFEM_BUILD_DIR) -lmfem -I/usr/local/include -lrt 

stltfd: stltfd.cpp MyTools.cpp
	g++ -g -o ../stltfd  -std=c++11 -I$(MFEM_DIR) stltfd.cpp MyTools.cpp -L$(MFEM_BUILD_DIR) -lmfem -I/usr/local/include -L/usr/lib/libgmsh.so -lgmsh -lrt 


cstl: cstl.cpp 
	g++ -g -o ../cstl  -std=c++11 -I$(MFEM_DIR) cstl.cpp -L$(MFEM_BUILD_DIR) -lmfem -I/usr/local/include -L/usr/lib/libgmsh.so -lgmsh -lrt 
