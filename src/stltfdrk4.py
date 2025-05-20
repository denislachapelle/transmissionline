# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import scipy.io as sio
import matplotlib.pyplot as plt
import math as mt


            
#Constants for the telegrapher’s equation for RG-58, 50 ohm.
L = 250e-9  # Inductance per unit length
C = 100.0e-12 # Capacitance per unit length
R = 220e-3  # Resistance per unit length
G = 1.0e-9  # Conductance per unit length

lenght = 100 #100 metre.
nbrSeg = 1000  #number of segment.
h = lenght/nbrSeg
deltaT = 0.01e-9
endTime = 500e-9
Time = 0.0

 
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
    return Dv.tocsr()

def create_Di_matrix(nbrSeg, C, h):
    nrow = nbrSeg + 1
    ncol = nbrSeg
    Di = lil_matrix((nrow, ncol))
    for i in range(1, nrow - 1):
        Di[i, i] = 1 / (C * h)
        Di[i, i - 1] = -1 / (C * h)
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
BM = createBlockMatrix(nbrSeg, Gv, Di, Dv, Ri)

x = np.zeros((2*nbrSeg+1, 1))

plotTime = endTime/20
plotCount = 0



while(Time<endTime):
    
    x[0, 0]= SourceFunction(Time)

    # Implementing the 4th order Runge-Kutta step here
    k1 = BM @ x
    k2 = BM @ ((deltaT/2) * k1 + x)
    k3 = BM @ ((deltaT/2) * k2 + x)
    k4 = BM @ (deltaT * k3 + x)

    # Final update for x
    x = x + deltaT * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    if Time > plotTime * plotCount:
        print(Time)

        #plot voltage    
        val=x[:nbrSeg+1]
        d = np.linspace(0, lenght, len(val)) 
        plt.figure(1)
        plt.plot(d, val)

        #plot current
        val=x[nbrSeg+1:]
        d = np.linspace(0, lenght, len(val)) 
        plt.figure(2)
        plt.plot(d, val)

        plotCount = plotCount + 1


#   x = Step(x, Time, deltaT)
    Time = Time + deltaT

plt.show()

