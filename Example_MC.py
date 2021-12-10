"""
Examples - Criteria for number of runs in stochastic simulations - Case MC simulations
August 2021
Created by jdmolina
"""

from SS_MC import T_Est, p_cal
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
#from sys import exit

############################################################
###################  Examples  #############################
############################################################    

### Example 1 ### 
def Gen1 (T):
    """
    This function generate values ​​of f(X) from Monte Carlo simulation
    """
    X = np.zeros((T, 3))
    for i in range(T): 
        np.random.seed(i) # The seed is fixing per row of the matrix
        X[i, :] = ss.norm.rvs(loc=3. , size=3)
    return X
    
def func1 (X):
    """
    This function calculates the functional g, 
    for a set of generated values of f(X)
    """
    g = np.zeros( X.shape[0] )    
    for i in range( X.shape[0] ):
        g[i] = sum(X[i, :])
    return g

## Results - T^* estimation
Res1 = T_Est(p=3, Gen=Gen1, func=func1) # Results from T* estimation
Tst1 = Res1[0] # 2,335,380
g1 = Res1[1]
CVI1 = Res1[2]
TI1 = Res1[3]
CV1 = Res1[4]
T1 = Res1[5]
SciNot1 = Res1[6] # (9.0, 0) , i.e. mu = 9.0 * 10**0

## Graph
plt.plot(T1[:-1] , CV1, '-o', color='b')
plt.plot(TI1 , CVI1, '*', color='k')
plt.axhline(y = np.sqrt(3)/9 , color='r' )
plt.xlabel('T')
plt.ylabel('CV estimation')
plt.savefig('T_CV_E1', dpi=300, format='jpeg')
## mantissa's estimation
hT = np.mean( g1 )  # 9.000309265206715
# Absolute difference between the real value and the estimation
mu = 9.0
dif_abs = abs(mu - hT) # 0.00030926520671492597  <  0.5 * 10^{-(p-1)} = 0.005


## Results - P calculation
T = 1e6
g_X = func1( Gen1( int(T) ) )
p_cal(g_X = g_X) 

### Example 2 ### 
def Gen2 (T):
    """
    This function generate values ​​of f(X) from Monte Carlo simulation
    """
    X = np.zeros((T, 4))
    for i in range(T): 
        np.random.seed(i) # The seed is fixing per row of the matrix
        X[i, :] = ss.expon.rvs(size=4)
    return X
    
def func2 (X):
    """
    This function calculates the functional g, 
    for a set of generated values of f(X)
    """
    g = np.zeros( X.shape[0] )    
    for i in range( X.shape[0] ):
        g[i] = np.mean(X[i, :])
    return g

Res2 = T_Est(p=3,Gen=Gen2, func=func2) # Results from T* estimation
Tst2 = Res2[0] # 15,895,850
g2 = Res2[1]
CVI2 = Res2[2]
TI2 = Res2[3]
CV2 = Res2[4]
T2 = Res2[5]
SciNot2 = Res2[6] # (1.0, 0) , i.e. mu = 1.0 * 10**0

## Graph
plt.plot(TI2 , CVI2, '*', color='k')
plt.plot(T2[:-1] , CV2, '-o', color='b')
plt.axhline(y = 0.5 , color='r' )
plt.xlabel('T')
plt.ylabel('CV estimation')
plt.savefig('T_CV_E2', dpi=300, format='jpeg')
## mantissa's estimation
hT = np.mean( g2 )  #  1.0001014643584167
# Absolute difference between the real value and the estimation
mu = 1.0
dif_abs = abs(mu - hT) # 8.674584170298427e-05  <  0.5 * 10^{-p} = 0.005

## Results - P calculation
T = 5e4
g_X = func2( Gen2( int(T) ) )
p_cal(g_X = g_X)
