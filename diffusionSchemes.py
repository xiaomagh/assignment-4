# Numerical schemes for simulating diffusion for outer code diffusion.py

from __future__ import absolute_import, division, print_function
import numpy as np

# The linear algebra package for BTCS (for solving the matrix equation)
import scipy.linalg as la

def FTCS(phiOld, d, nt):
    "Diffusion of profile in phiOld using FTCS using non-dimensional"
    "diffusion coeffient, d"
    
    nx = len(phiOld)
    
    # new time-step array for phi
    phi = phiOld.copy()
    
    M = np.zeros([nx,nx])
    # Zero gradient boundery conditions
    M[0,0] = 1.
    M[0,1] = -1.
    M[-1,-1] = 1.
    M[-1,-2] = -1.    
    for i in xrange(1,nx-1):
        M[i,i-1] = d
        M[i,i] = 1 - 2*d
        M[i,i+1] = d
    
    #FCTS for all time steps
    for it in xrange(int(nt)):
        
        phi = np.dot(M,phi)
        
    return phi
        
                  
                  
def BTCS(phi, d, nt):
    "Diffusion of profile in phi using BTCS using non-dimensional"
    "diffusion coefficient, d assuming fixed value boundary conditions"
    nx = len(phi)
    
    #array representing BTCS
    M = np.zeros([nx,nx])
    # Zero gradient boundery conditions
    M[0,0] = 1.
    M[0,1] = -1.
    M[-1,-1] = 1.
    M[-1,-2] = -1.
    for i in xrange(1,nx-1):
        M[i,i-1] = -d
        M[i,i] = 1 + 2*d
        M[i,i+1] = -d
        
    #BTCS for all time steps
    for it in xrange(int(nt)):
        #RHS for zero gradient boundary conditions
        phi[0] = 0
        phi[-1] = 0
        
        phi = la.solve(M,phi)
        
    return phi