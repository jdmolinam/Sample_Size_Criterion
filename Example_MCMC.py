"""
Example - Criteria for number of runs in stochastic simulations - Case MCMC simulations
August 2021
Created by jdmolina
"""

from SS_MCMC import T_Est_MCMC, p_cal_MCMC
import numpy as np
#import scipy.stats as ss
import scipy.integrate as si
import numpy.random as nr
#from math import log10, floor, ceil
import matplotlib.pyplot as plt
#from sys import exit
import pytwalk # This module must be install firts. See: https://www.cimat.mx/~jac/twalk/
from numpy import ones, zeros, cumsum, shape, mat, cov, mean, matrix, sqrt
from numpy import exp, log, sum, pi, savetxt, loadtxt, array


############################################################
###################  Example  ##############################
############################################################    

############################################################
###### Definition of the structure of the example ##########
###### and the stabilization point of the chain ############
############################################################

### Lotka volterra equations ###
def SistEc (w, t):
    """
    This function define the Lotka volterra equations
    w : is a vector of the state variables
        w = [u1 , u2]
    """
    u1 , u2 = w
    # Definition of the system, F = (u1' , u2'):
    F = [u1*(1. - u2) , 
         u2*(u1 -1.)]
    return F

### Observations Simulation ###
# True value of the initial conditions
u1_0_ob = 0.5 # The units are thousands of specimens
u2_0_ob = 2. # The units are thousands of specimens
w0_ob = [u1_0_ob , u2_0_ob]
# Time of the observations
t_ob = np.array([0., 0.75, 1.5, 2.25, 3.0, 3.75])
# system solution
u1_ob = si.odeint(SistEc , w0_ob, t_ob)[: , 0]
sigma = max(abs(u1_ob))*0.1 # 0.11512233018166435
s2 = sigma**2
nr.seed(123) # the seed is fixed
Y = u1_ob + nr.normal(scale=sigma, size=len(t_ob))

### Simulation of the posterior distribution using the t-walk ###
# Support Definition
def LotVolSupp (theta): # theta[0]=u10, theta[1]=u20
    rt = True
    rt &= (0.3 <= theta[0] <= 0.7)
    rt &= (1.8 <= theta[1] <= 2.2)
    return rt

# Initial values definition
def LotVolInit (): 
    """
    This function generates initial values
    from the prior distribution
    """
    u10 = np.random.uniform(0.3 , 0.7)
    u20 = np.random.uniform(1.8 , 2.2)
    return np.array([u10 , u20])

## Definition of energy function
def LotVolU (theta):  
    """
    This function calculates the energy of the objective posterior distribution, 
    evaluated in candidates u10, u20.
    """  
    u = si.odeint(func=SistEc , y0=theta, t=t_ob)[: , 0]
    
    return sum( (Y - u)**2 ) / (2*s2)

### Run the t-walk ###
# t-walk object definition
LotVol = pytwalk.pytwalk( n=2, U=LotVolU, Supp=LotVolSupp)
## Order for run the t-walk
#LotVol.Run( T=10000, x0=LotVolInit(), xp0=LotVolInit())
# Chain analysis
#LotVol.Ana() # It's recommendable a burn-in of 500
# LotVol.Ana(start=500)


def Gen3 (T):
    """
    This function generate values ​​of f(X) from MCMC simulation
    """
    return LotVol.Run( T = int( T ), x0=LotVolInit(), xp0=LotVolInit())
    
def func3 (X, start):
    """
    This function calculates the functional g, 
    for a set of generated values of f(X)
    """
    g = LotVol.Output[start: , 0:1] # g has matricial shape, with one column
    return g

### Results - T^* estimation ###
Res = T_Est_MCMC(p=2, Gen=Gen3, func=func3, start=5000) # Results from T* estimation
Tst = Res[0] # 942,270
g = Res[1]
iat = Res[2] # 51.7989954598603
CVI = Res[3]
TI = Res[4]
CV = Res[5]
T = Res[6]
SciNot = Res[7] # # (4.4, -1) , i.e. mu = 4.4 * 10**{-1}

# Final estimation - Cv
CvF = np.sqrt( iat * np.var(g) ) / np.mean(g) # 1.209973337521288

### Results - p calculation ###
start = 5000
T = 2.5e5
g_X = func3( Gen3(T+start), start )
p_cal_MCMC(g_X = g_X) 

## Graphs posterior distributions
LotVol.Run( T= int(T + start), x0=LotVolInit(), xp0=LotVolInit())
u10 = LotVol.Output[start: , 0] 
u20 = LotVol.Output[start: , 1]
np.savetxt('u10_25-4k', u10)
np.savetxt('u20_25-4k', u20)
# Histograms
plt.hist(u10, density=True)
plt.axvline(x=0.5, color='red')
plt.xlabel(r'$u_1 ^0$')
plt.ylabel("Density")
plt.savefig('Post_u10.jpeg', dpi=300, format='jpeg')
plt.figure()
plt.hist(u20, density=True)
plt.axvline(x=2., color='red')
plt.xlabel(r'$u_2 ^0$')
plt.ylabel("Density")
plt.savefig('Post_u20.jpeg', dpi=300, format='jpeg')
