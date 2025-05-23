# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from scipy.sparse import lil_matrix
import scipy.io as sio
import matplotlib.pyplot as plt
import math as mt


            
#Constants for the telegrapher’s equation for RG-58, 50 ohm.
L = 250e-9  # Inductance per unit length
C = 100.0e-12 # Capacitance per unit length
R = 220e-3  # Resistance per unit length
G = 1.0e-9  # Conductance per unit length

Rs = 50

lenght = 100 #100 metre.
nbrSeg = 1000  #number of segment.
h = lenght/nbrSeg
deltaT = 0.01e-9
endTime = 1500e-9
Time = 0.0

def PrintCSR(A, filename):
    with open(filename, "w") as f:
       for i, j in zip(*A.nonzero()):
          f.write(f"({i}, {j})\t{A[i, j]}\n")
 
# generate a time dependent signal
def SourceFunction(t):
      # gaussian pulse of tw wide centered at tc.*/
      tw = 20e-9
      tc = 50e-9
      if(t<2*tc):
         return 4.0 * t/(2.0*tc)*(1-t/(2.0*tc)) * mt.exp(-mt.pow(((t-tc)/tw), 2.0))
      else:
         return 0.0
     
def create_Dv_matrix(nbrSeg, L, h):
    nrow = nbrSeg
    ncol = nbrSeg + 1
    Dv = lil_matrix((nrow, ncol))
    for i in range(nrow):
        Dv[i, i] = -1 / (L * h)
        Dv[i, i + 1] = 1 / (L * h)
    
    PrintCSR(Dv.tocsr(), "out/dv.txt")
    return Dv.tocsr()

def create_Di_matrix(nbrSeg, C, h):
    nrow = nbrSeg + 1
    ncol = nbrSeg
    Di = lil_matrix((nrow, ncol))
    for i in range(0, nrow - 1):
        Di[i, i] = 1 / (C * h)
        if i >= 0:
           Di[i, i - 1] = -1 / (C * h)

    #we should complete the last row which compute the last node voltage Vx.
    
    PrintCSR(Di.tocsr(), "out/di.txt")
    return Di.tocsr()


def create_Ri_matrix(nbrSeg, R, L):
    size = nbrSeg
    val = -R / L
    Ri = lil_matrix((size, size))
    # Create a diagonal sparse matrix
    for i in range(size):
        Ri[i, i] = val
        
    return Ri.tocsr()

def create_Gv_matrix(nbrSeg, G, C):
    size = nbrSeg + 1
    val = -G / C
    Gv = lil_matrix((size, size))
    # Create a diagonal sparse matrix
    for i in range(size):
        Gv[i, i] = val
        
    return Gv.tocsr()

def createBlockMatrix(nbrSeg, smGv, smDi, smDv, smRi):
    
    # Initialize fdBlockOperator (a sparse matrix of size 2*(nbrSeg+1) x 2*(nbrSeg+1))
    BM = lil_matrix((2 * nbrSeg + 1, 2 * nbrSeg + 1))

    # Insert each block into fdBlockOperator
    BM[:nbrSeg+1, :nbrSeg+1] = smGv 
    BM[:nbrSeg+1, nbrSeg+1:] = smDi
    BM[nbrSeg+1:, :nbrSeg+1] = smDv
    BM[nbrSeg+1:, nbrSeg+1:] = smRi

    #Print BM as a Matlab format matrix
    sio.savemat('BM.txt', {'BM': BM.toarray()})

    return BM.tocsr()

Di = create_Di_matrix(nbrSeg, C, h)
Dv = create_Dv_matrix(nbrSeg, L, h)
Ri = create_Ri_matrix(nbrSeg, R, L)
Gv = create_Gv_matrix(nbrSeg, G, C)
BM1 = createBlockMatrix(nbrSeg, Gv, Di, Dv, Ri)
SM = create_Source_matrix(2*nbrSeg+2, Rs, C, h)
oneCol = csr_matrix((2*nbrSeg+1, 1))
BM2 = hstack( [oneCol, BM1])
BM2[0, 0] = 1/(C*h*Rs)
BM2[0, 1] = BM2[0, 1] - 1/(C*h*Rs)
oneRow = csr_matrix((1, 2*nbrSeg+2))
BM3 = vstack([oneRow, BM2])

x = np.zeros((2*nbrSeg+2, 1))

plotTime = endTime/15
plotCount = 1



while(Time<endTime):
    
    x[0, 0]= SourceFunction(Time)

    # Implementing the 4th order Runge-Kutta step here
    k1 = BM3 @ x
    k2 = BM3 @ ((deltaT/2) * k1 + x)
    k3 = BM3 @ ((deltaT/2) * k2 + x)
    k4 = BM3 @ (deltaT * k3 + x)

    # Final update for x
    x = x + deltaT * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    if Time > plotTime * plotCount:
        print(Time)

        #plot voltage    
        val=x[1:nbrSeg+2]
        d = np.linspace(0, lenght, len(val)) 
        plt.figure(plotCount)
        plt.plot(d, val)
        plt.grid()
        plotCount = plotCount + 1
    
    Time = Time + deltaT

plt.show()

