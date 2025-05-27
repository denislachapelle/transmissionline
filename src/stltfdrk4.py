# -*- coding: utf-8 -*-
"""
stltfdrk4.py stands for 
"Single Transmission Line Transient Finite Difference Runge Kutta 4"

This program use finite difference and Runge Kutta 4 time stepping 
to simulate a transmission line propagating signal. It is based on
the telegrapher equations.

written by Denis Lachapelle, May 2025.
denislachapelle2003_gmail.com
"""

import math as mt
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

Rs = 50 #source resistance.
Rl = 1  #load resistance.

lenght = 100 #100 metre.
nbrSeg = 1000  #number of segment.
h = lenght/nbrSeg   #segment lenght.
deltaT = 0.01e-9    #time step.
endTime = 1500e-9   #tiem in second.
Time = 0.0          #time counter starts at 0.

def PrintCSR(A, filename):
    with open(filename, "w") as f:
       for i, j in zip(*A.nonzero()):
          f.write(f"({i}, {j})\t{A[i, j]}\n")
 
#This function is a tool to simplify in selecting the desired source function. 
def SourceFunction(t):
    return SourceFunctionStep(t)

# generate a time dependent signal
def SourceFunctionGaussianPulse(t):
      # gaussian pulse of tw wide centered at tc.*/
      tw = 20e-9
      tc = 50e-9
      if(t<2*tc):
         return 4.0 * t/(2.0*tc)*(1-t/(2.0*tc)) * mt.exp(-mt.pow(((t-tc)/tw), 2.0))
      else:
         return 0.0
     
def SourceFunctionGaussianSine(t):
      return np.sin(2*mt.pi*13e6*t)
      
def SourceFunctionStep(t):
      #pulse.
      tau = 10e-9
      return 1.0 - mt.exp(-t/tau)

'''
the Dv matrix derive the voltage to compute the current,
so it has nbrSeg row and nbrSeg+1 column.
'''
def create_Dv_matrix(nbrSeg, L, h):
    nrow = nbrSeg
    ncol = nbrSeg + 1
    Dv = lil_matrix((nrow, ncol))
    for i in range(nrow):
        Dv[i, i] = -1 / (L * h)
        Dv[i, i + 1] = 1 / (L * h)
    
    return Dv

'''
   the Di matrix does not include the boundary conditions (the two if)
   applied on the left and right side, they will have to be condidered
   in the loop stepping in time.
'''
def create_Di_matrix(nbrSeg, C, h):
    nrow = nbrSeg + 1
    ncol = nbrSeg
    Di = lil_matrix((nrow, ncol))
    for i in range(0, nrow):
        if i < ncol:
            Di[i, i] = 1 / (C * h)
        if i > 0:
            Di[i, i - 1] = -1 / (C * h)
   
    return Di


def create_Ri_matrix(nbrSeg, R, L):
    size = nbrSeg
    val = -R / L
    Ri = lil_matrix((size, size))
    # Create a diagonal sparse matrix
    for i in range(size):
        Ri[i, i] = val
        
    return Ri

def create_Gv_matrix(nbrSeg, G, C):
    size = nbrSeg + 1
    val = -G / C
    Gv = lil_matrix((size, size))
    # Create a diagonal sparse matrix
    for i in range(size):
        Gv[i, i] = val
        
    return Gv


def createBlockMatrix(nbrSeg, smGv, smDi, smDv, smRi):
    
    # Initialize BM (a sparse matrix of size 2*(nbrSeg+1) x 2*(nbrSeg+1))
    BM = lil_matrix((2 * nbrSeg + 2, 2 * nbrSeg + 2))

    # Insert each block into Block Matrix
    BM[1:nbrSeg+2, 1:nbrSeg+2] = smGv 
    BM[1:nbrSeg+2, nbrSeg+2:] = smDi
    BM[nbrSeg+2:, 1:nbrSeg+2] = smDv
    BM[nbrSeg+2:, nbrSeg+2:] = smRi

    #add boundary condition for Vs and Rs.
    BM[1, 0] = BM[1, 0] + 1/(Rs*C*h)
    BM[1, 1] = BM[1, 1] - 1/(Rs*C*h)
    
    #add boundary condition for Rl.
    BM[nbrSeg+1, nbrSeg+1] = BM[nbrSeg+1, nbrSeg+1] - 1/(Rl*C*h)

    #Print BM as a Matlab format matrix
    sio.savemat('BM.txt', {'BM': BM.toarray()})

    return BM.tocsr()

Di = create_Di_matrix(nbrSeg, C, h)
Dv = create_Dv_matrix(nbrSeg, L, h)
Ri = create_Ri_matrix(nbrSeg, R, L)
Gv = create_Gv_matrix(nbrSeg, G, C)
BM = createBlockMatrix(nbrSeg, Gv, Di, Dv, Ri)

x = np.zeros((2*nbrSeg+2, 1))

nbrPlot=5
plotTime = endTime/nbrPlot
plotCount = 1
plt.figure(1)

while(Time<=endTime):
    
    #apply the left boundary condition which is a voltage source with Rs in séries.
    #x[0, 0] = x[0, 0] + deltaT*(SourceFunction(Time) - x[0, 0])/(Rs*C*h)

    #apply the right boundary condition whichis a load Rl.
    #x[nbrSeg, 0] = x[nbrSeg, 0] - deltaT*x[nbrSeg, 0]/(Rl*C*h)

    # Implementing the 4th order Runge-Kutta step here
    x[0, 0] = SourceFunction(Time)
    k1 = BM @ x
    x[0, 0] = SourceFunction(Time+deltaT/2)
    k2 = BM @ ((deltaT/2) * k1 + x)
    x[0, 0] = SourceFunction(Time+deltaT/2)
    k3 = BM @ ((deltaT/2) * k2 + x)
    x[0, 0] = SourceFunction(Time+deltaT)
    k4 = BM @ (deltaT * k3 + x)

    # Final update for x
    x = x + deltaT * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    Time = Time + deltaT

    #plot the voltage at each plotTime second.
    if Time >= plotTime * plotCount:
        print(Time)
        #plot voltage only.   
        val=x[1:nbrSeg+2]
        print(np.max(np.abs(val)))
        d = np.linspace(0, lenght, len(val)) 
        plt.subplot(nbrPlot, 1, plotCount)
        plt.plot(d, val)
        plt.ylim(-1, 1)
        plt.grid()
        plotCount = plotCount + 1
    
   

plt.show()

