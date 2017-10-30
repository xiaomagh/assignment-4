#!/usr/bin/python

# Outer code for setting up the diffusion problem on a uniform
# grid and calling the function to perform the diffusion and plot.

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np

# read in all the linear advection schemes, initial conditions and other
# code associated with this application
execfile("diffusionSchemes.py")
execfile("diagnostics.py")
execfile("initialConditions.py")
#execfile("initialConditions1.py")

def main():
    # Insert the rest of the code for main here
    "Diffuse a squareWave  between squareWaveMin and squareWaveMax on a"
    "domain between x = xmin and x = xmax spilit over nx spatial steps"
    "with diffusion coeffient K, time step dt for nt time steps"
    
    # Parameters
    xmin = 0.
    xmax = 1.
    nx = 41
    nt = 50
    dt = 0.1
    K = 1e-3
    squareWaveMin = 0.4
    squareWaveMax = 0.6
    
    # Derived parameters
    dx = (xmax - xmin)/(nx - 1)
    d = K*dt/dx**2   #Non-dimensional diffusion coeffient
    print("non-dimensional diffusion coefficient = ", d)
    print("dx = ", dx, "dt = ", dt, "nt = ", nt)
    print("end time = ", nt*dt)
    
    # spatial point for plotting and for defining initial conditions
    x = np.zeros(nx)
    for j in xrange(nx):
        x[j] = xmin + j*dx
        
    print('x=',x)
    
    # Initial conditions
    phiOld = squareWave(x, squareWaveMin, squareWaveMax)
    # Analytic solution (of square wave profile in an infinite domain)
    phiAnalytic = analyticErf(x, K*dt*nt, squareWaveMin, squareWaveMax)
    
    # Diffusion using FTCS and BTCS
    phiFTCS = FTCS(phiOld.copy(), d, nt)
    phiBTCS = BTCS(phiOld.copy(), d, nt)
    
    # Calculate and print out error norms
    FTCSErrorNorm, FTCSError = L2ErrorNorm(phiFTCS, phiAnalytic)
    BTCSErrorNorm, BTCSError = L2ErrorNorm(phiBTCS, phiAnalytic)
    
    print (FTCSErrorNorm,BTCSErrorNorm)
    
    # plot the solutions
    font = {'size'  :15}
    plt.rc('font', **font)
    plt.figure(1)
    plt.clf()
    plt.ion()
    plt.plot(x, phiOld, label='Initial', color='black')
    plt.plot(x, phiAnalytic, label='Analytic', color='black', linestyle='-', linewidth=2)
    plt.plot(x, phiFTCS, label='FTCS', color='blue')
    plt.plot(x, phiBTCS, label='BTCS', color='red')
    plt.axhline(0, linestyle=':', color='black')
    plt.ylim([0,1.0])
    plt.legend(bbox_to_anchor=(1.1, 1))
    plt.xlabel('$x$')
    plt.savefig('FTCS_BTCS.pdf')
    
    font = {'size'  :12}
    plt.rc('font', **font)
    plt.figure(2)
    plt.clf()
    plt.ion()
    plt.plot(x[1:40], FTCSError, label='FTCS Error', color='blue')
    plt.plot(x[1:40], BTCSError, label='BTCS Error', color='red')
    plt.legend(bbox_to_anchor=(1.1, 1))
    plt.xlabel('$x$')
    plt.savefig('ERROR.pdf')
       
   
   # Run FTCS for a sufficiently long duration
    phiAnalytic1 = analyticErf(x, K*dt*500, squareWaveMin, squareWaveMax)
    phiFTCS1 = FTCS(phiOld.copy(), d, 500)
    phiAnalytic2 = analyticErf(x, K*dt*2000, squareWaveMin, squareWaveMax)
    phiFTCS2 = FTCS(phiOld.copy(), d, 2000)
    FTCSErrorNorm1, FTCSError1 = L2ErrorNorm(phiFTCS1, phiAnalytic1)
    FTCSErrorNorm2, FTCSError2 = L2ErrorNorm(phiFTCS2, phiAnalytic2)
    
    font = {'size'  :12}
    plt.rc('font', **font)
    plt.figure(3)
    plt.clf()
    plt.ion()
    plt.plot(x, phiOld, label='Initial', color='black')
    plt.plot(x, phiAnalytic1, label='Analytic', color='black', linestyle='-', linewidth=2)
    plt.plot(x, phiFTCS1, label='FTCS nt=500', color='blue')
    plt.axhline(0, linestyle=':', color='black')
    plt.ylim([0,1.0])
    plt.legend(bbox_to_anchor=(1.1, 1))
    plt.xlabel('$x$')
    plt.savefig('FTCSlong1.pdf')
    
    font = {'size'  :12}
    plt.rc('font', **font)
    plt.figure(4)
    plt.clf()
    plt.ion()
    plt.plot(x, phiOld, label='Initial', color='black')
    plt.plot(x, phiAnalytic2, label='Analytic', color='black', linestyle='-', linewidth=2)
    plt.plot(x, phiFTCS2, label='FTCS nt=2000', color='blue')
    plt.axhline(0, linestyle=':', color='black')
    plt.ylim([0,1.0])
    plt.legend(bbox_to_anchor=(1.1, 1))
    plt.xlabel('$x$')
    plt.savefig('FTCSlong2.pdf')
    
    font = {'size'  :12}
    plt.rc('font', **font)
    plt.figure(5)
    plt.clf()
    plt.ion()
    plt.plot(x[1:40], FTCSError, label='Error nt=40', color='blue')
    plt.plot(x[1:40], FTCSError1, label='Error nt=500', color='red')
    plt.plot(x[1:40], FTCSError2, label='Error nt=2000', color='green')
    plt.legend(bbox_to_anchor=(0.75, 0.6))
    plt.xlabel('$x$')
    plt.savefig('ERRORlong.pdf')
    
   
   #The experiment to test the stability
    Fpeak = np.zeros(14)
    Bpeak = np.zeros(14)
    a = np.zeros(13)
    b = np.zeros(13)
    c = np.zeros(14)
    dtt = np.zeros(14)  
    c = [40,36,32,30,28,25,24,20,18,16,12,10,8,4]
    for n in xrange(0,14):    
        dtt[n] = 4.0/c[n]
        dd = K*dtt[n]/dx**2
        phiF = FTCS(phiOld.copy(), dd, n)
        phiB = BTCS(phiOld.copy(), dd, n)
        Fpeak[n] = phiF[20]
        Bpeak[n] = phiB[20]
        c[n] = (dx**2)/(dtt[n])
    for i in xrange(0,13):
        a[i] = Fpeak[i+1] / Fpeak[i]
        b[i] = Bpeak[i+1] / Bpeak[i]
    
    print (a,b,dtt)

    font = {'size'  :12}
    plt.rc('font', **font)
    plt.figure(6)
    plt.clf()
    plt.ion()
    plt.plot(dtt, Fpeak, label='FTCS with dt', color='blue')
    plt.plot(dtt, Bpeak, label='BTCS with dt', color='red')
    plt.ylim([-4.0,10])
    plt.legend(bbox_to_anchor=(1.1, 1))
    plt.xlabel('$dt/s$')
    plt.savefig('stability.pdf')
    plt.figure(10)
    plt.clf()
    plt.ion()
    plt.plot(c,Fpeak,color='blue')
    plt.plot(c,Bpeak,color='red')
    plt.axhline(0, linestyle=':', color='black')
    plt.ylim([-4.0,10])
    plt.legend(bbox_to_anchor=(1.1, 1))
    plt.xlabel('$dx^2/dt$')
    plt.ylabel('$phi$')
    plt.savefig('stability02.pdf')
       
   
   #test the order of convergence
    dxx = np.zeros(3)
    dtt = np.zeros(3)
    l2 = np.zeros(3)
    n1 = np.zeros(2)
    n2 = np.zeros(2)
    for i in xrange(0,3):
        ntt = 40 * (4**i)
        dtt[i] = 4.0/ntt
        dxx[i] = np.sqrt((K*dtt[i])/(K*0.1/0.025**2)) #make sure d stays the same
        nxx = int(1./dxx[i]) + 1
        xx = np.zeros(nxx)
        for j in xrange(nxx):
            xx[j] = xmin + j*dxx[i]           
        phiOld = squareWave(xx, squareWaveMin, squareWaveMax)
        phiF = FTCS(phiOld.copy(), K*0.1/0.025**2, ntt)
        Analytic = analyticErf(xx, K*4.0, squareWaveMin, squareWaveMax)
        l2[i], error = L2ErrorNorm(phiF, Analytic)
    print (l2,dxx,dtt)
    #calculate the order
    for i in xrange(0,2):
        n1[i] = (np.log(l2[i]) - np.log(l2[i+1]))/(np.log(dxx[i]) - np.log(dxx[i+1]))
        n2[i] = (np.log(l2[i]) - np.log(l2[i+1]))/(np.log(dtt[i]) - np.log(dtt[i+1]))     
    print (n1,n2)
    
    font = {'size'  :12}
    plt.rc('font', **font)
    plt.figure(7)
    plt.clf()
    plt.ion()
    plt.plot(dxx, l2)
    plt.axhline(0, linestyle=':', color='black')
    plt.legend(bbox_to_anchor=(1.1, 1))
    plt.xlabel('dx')
    plt.savefig('convergence.pdf') 
    


main()


