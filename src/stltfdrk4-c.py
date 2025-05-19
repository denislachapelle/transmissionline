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
endTime = 100e-9
Time = 0.0

 
# generate a time dependent signal
def SourceFunction(t):
      #return 1.0
      # gaussian pulse of tw wide centered at tc.*/
      tw = 20e-9
      tc = 50e-9
      if(t<2*tc):
         return 4.0 * t/(2.0*tc)*(1-t/(2.0*tc)) * mt.exp(-mt.pow(((t-tc)/tw), 2.0))
      else:
         return 0.0
     
def create_Dv_matrix(nbrSeg, L, h):
    nrow = nbrSeg
    ncol = nbrSeg+1
    val = -1 / (L * h)

    # Use LIL format for efficient row-wise construction
    Dv = lil_matrix((nrow, ncol))

    for i in range(nrow):
        if i < ncol:
            Dv[i, i+1] = val
        if i >= 0:
            Dv[i, i] = -val     

    # Convert to CSR format for efficient computation
    return Dv.tocsr()


def create_Di_matrix(nbrSeg, C, h):
    nrow = nbrSeg + 1
    ncol = nbrSeg
    val = -1 / (C * h)

    Di = lil_matrix((nrow, ncol))

    for i in range(nrow):
        if i+1 < ncol:
           Di[i, i+1] = val
        if i < ncol:
            Di[i, i] = -val

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

Vx = np.zeros((nbrSeg+1, 1))
Ix = np.zeros((nbrSeg, 1))



def Step1(x, t, deltaT):
        # Implementing the 4th order Runge-Kutta step here
        k1 = BM @ x
        #k2 = BM @ ((deltaT/2) * k1 + x)
        #k3 = BM @ ((deltaT/2) * k2 + x)
        #k4 = BM @ (deltaT * k3 + x)

        # Final update for x
        #return x + deltaT * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        return x + deltaT * k1


while(Time<endTime):
    Vx[0, 0]= SourceFunction(Time)

    dvdt1 = Gv @ Vx + Di @ Ix
    didt1 = Dv @ Vx + Ri @ Ix

    Vx = Vx + deltaT * dvdt1
    Ix = Ix + deltaT * didt1

#   x = Step(x, Time, deltaT)
    Time = Time + deltaT

plt.figure(1)
plt.plot(Vx)
plt.figure(2)
plt.plot(Ix)
plt.show()

