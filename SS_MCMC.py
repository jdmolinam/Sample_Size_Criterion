"""
Criteria for number of runs in stochastic simulations - Case MCMC simulations
November 2020
Created by jdmolina
"""

############################################################
#####################   Modules  ###########################
############################################################

import numpy as np
import scipy.stats as ss
import scipy.integrate as si
import numpy.random as nr
from math import log10, floor, ceil
import matplotlib.pyplot as plt
from sys import exit
#import pytwalk
from numpy import ones, zeros, cumsum, shape, mat, cov, mean, matrix, sqrt
from numpy import exp, log, sum, pi, savetxt, loadtxt, array


############################################################
############### Auxiliar Functions  ########################
############################################################

def mu_B (X):
    """
    This function calculate a preliminar estimation of mu, 
    for a sample stored in a vector X. 
    The estimation is based on the method proposed by Bland (2015).     
    """
    a = min(X)
    q1 = np.percentile(X , 25)
    m = np.median(X)
    q3 = np.percentile(X , 75)
    b = max(X)
    n = len(X)    
    ## Sample average's estimation
    # Lower boundary for the sample average
    IP = (a + q1 + m + q3)/4 + (4*b - a - q1 - m - q3)/(4*n)
    # Upper boundary for the sample average
    SP = (q1 + m + q3 + b)/4 + (4*a - q1 - m - q3 -b)/(4*n)
    X_bar = (IP + SP)/2
    return X_bar


def SigDigitsMatissaEsp( x, sig_digits=4):
    """
    x is a real number.
    Return the matissa m as a float, 1 <= m < 10 and exponent q as an int.
    """  
    tmp = np.format_float_scientific( x, precision=sig_digits-1, trim='0', exp_digits=3)
    return float(tmp[:-5]), int(tmp[-4:])


############################################################################################
#### Functions to calculate Integrated autocorrelation times of a time series 
## Taken from code of the python version of t-walk, christen and Fox (2010).

####  Calculates an autocovariance 2x2 matrix at lag l in column c of matrix Ser with T rows
####  The variances of each series are in the diagonal and the (auto)covariance in the off diag.
def AutoCov( Ser, c, la, T=0):
    if (T == 0):
        T = shape(Ser)[0]  ### Number of rows in the matrix (sample size)

    return cov( Ser[0:(T-1-la), c], Ser[la:(T-1), c], bias=1)
    
    
    

#### Calculates the autocorrelation from lag 0 to lag la of columns cols (list)
#### for matrix Ser
def AutoCorr( Ser, cols=0, la=1):
    T = shape(Ser)[0]  ### Number of rows in the matrix (sample size)

    ncols = shape(mat(cols))[1] ## Number of columns to analyse (parameters)

    #if ncols == 1:
    #    cols = [cols]
        
    ### Matrix to hold output
    Out = matrix(ones((la+1)*ncols)).reshape( la+1, ncols)
        
    for c in range(ncols):
        for l in range( 1, la+1):  
            Co = AutoCov( Ser, cols[c], l, T) 
            Out[l,c] = Co[0,1]/(sqrt(Co[0,0]*Co[1,1]))
    
    return Out
    

### Makes an upper band matrix of ones, to add the autocorrelation matrix
### gamma = auto[2*m+1,c]+auto[2*m+2,c] etc. 
### MakeSumMat(lag) * AutoCorr( Ser, cols=c, la=lag) to make the gamma matrix
def MakeSumMat(lag):
    rows = (lag)//2   ### Integer division!
    Out = mat(zeros([rows,lag], dtype=int))
    
    for i in range(rows): 
        Out[i,2*i] = 1
        Out[i,2*i+1] = 1
    
    return Out


### Finds the cutting time, when the gammas become negative
def Cutts(Gamma):
    cols = shape(Gamma)[1]
    rows = shape(Gamma)[0]
    Out = mat(zeros([1,cols], dtype=int))
    Stop = mat(zeros([1,cols], dtype=bool))
    
    if (rows == 1):
        return Out
        
    i = 0
    ###while (not(all(Stop)) & (i < (rows-1))):
    for i in range(rows-1):
        for j in range(cols):  ### while Gamma stays positive and decreasing
            if (((Gamma[i+1,j] > 0.0) & (Gamma[i+1,j] < Gamma[i,j])) & (not Stop[0,j])):
                Out[0,j] = i+1 ## the cutting time for colomn j is i+i
            else:
                Stop[0,j] = True
        i += 1
    
    
    return Out


####  Automatically find a maxlag for IAT calculations
def AutoMaxlag( Ser, c, rholimit=0.05, maxmaxlag=20000):
    Co = AutoCov( Ser, c, la=1)
    rho = Co[0,1]/Co[0,0]  ### lag one autocorrelation
    
    ### if autocorrelation is like exp(- lag/lam) then, for lag = 1
    lam = -1.0/log(abs(rho)) 
    
    ### Our initial guess for maxlag is 1.5 times lam (eg. three times the mean life)
    maxlag = int(floor(3.0*lam))+1
    
    ### We take 1% of lam to jump forward and look for the
    ### rholimit threshold
    jmp = int(ceil(0.01*lam)) + 1
    
    T = shape(Ser)[0]  ### Number of rows in the matrix (sample size)

    while ((abs(rho) > rholimit) & (maxlag < min(T//2,maxmaxlag))):
        Co = AutoCov( Ser, c, la=maxlag)
        rho = Co[0,1]/Co[0,0]
        maxlag = maxlag + jmp
        ###print("maxlag=", maxlag, "rho", abs(rho), "\n")
        
    maxlag = int(floor(1.3*maxlag));  #30% more
    
    if (maxlag >= min(T//2,maxmaxlag)): ###not enough data
        fixmaxlag = min(min( T//2, maxlag), maxmaxlag)
        print("AutoMaxlag: Warning: maxlag= %d > min(T//2,maxmaxlag=%d), fixing it to %d" % (maxlag, maxmaxlag, fixmaxlag))
        return fixmaxlag
    
    if (maxlag <= 1):
        fixmaxlag = 10
        print("AutoMaxlag: Warning: maxlag= %d ?!, fixing it to %d" % (maxlag, fixmaxlag))
        return fixmaxlag
        
    print("AutoMaxlag: maxlag= %d." % maxlag)
    return maxlag
    
    
### Find the IAT
def IAT( Ser, cols=-1,  maxlag=0, start=0, end=0):

    ncols = shape(mat(cols))[1] ## Number of columns to analyse (parameters)
    if ncols == 1:
        if (cols == -1):
            cols = shape(Ser)[1]-1 ### default = last column
        cols = [cols]
    
    if (end == 0):
        end = shape(Ser)[0]

    if (maxlag == 0):
        for c in cols:
            maxlag = max(maxlag, AutoMaxlag( Ser[start:end,:], c))

    #print("IAT: Maxlag=", maxlag)

    #Ga = MakeSumMat(maxlag) * AutoCorr( Ser[start:end,:], cols=cols, la=maxlag)
    
    Ga = mat(zeros((maxlag//2,ncols)))
    auto = AutoCorr( Ser[start:end,:], cols=cols, la=maxlag)
    
    ### Instead of producing the maxlag/2 X maxlag MakeSumMat matrix, we calculate the gammas like this
    for c in range(ncols):
        for i in range(maxlag//2):
            Ga[i,c] = auto[2*i,c]+auto[2*i+1,c]
    
    cut = Cutts(Ga)
    nrows = shape(Ga)[0]
        
    ncols = shape(cut)[1]
    Out = -1.0*mat(ones( [1,ncols] ))
    
    if any((cut+1) == nrows):
        print("IAT: Warning: Not enough lag to calculate IAT")
    
    for c in range(ncols):
        for i in range(cut[0,c]+1):
            Out[0,c] += 2*Ga[i,c]
    
    return Out

############################################################################################


############################################################
############### Principal Functions  ########################
############################################################

def T_Est_MCMC(p, Gen, func, start):
    """
    This function allows to estimate T^*, for a MCMC sample, for a p given precision, 
    when the mechanism to generate values of the sample 
    and to calculate the functional are known.
    p : determines the precision in the mantissa's estimation
    start: is the point for the burn-in
    Gen : function that allows to generate values of the sample of interest, it must return an array of dim T X m (T is the number of simulations and m the dimension of the variable) 
    func : function that calculates the functional of the sample, must return an array of dimension T X 1, i.e., a matrix with one column. 
    """
    ### initial stage ###            
    CVI_trace = np.array([]) # this object will save the record of the initial CV's estimations
    TI_trace = np.array([]) # this object will save the record of the initial T*'s estimations
    tau = 1/4
    T = int( (8*tau*10)**2 )
    TI_trace = np.append(TI_trace, T)
    g = func( Gen(T + start), start )
    iat = IAT(g)[0,0] # the function return a matrix 1X1, we take only the scalar
    CV = np.sqrt( iat * np.var(g) ) / mu_B(g)  # CV's preliminar estimation 
    CVI_trace = np.append(CVI_trace, CV)
    
    while (CV > tau and tau <= 4):
        tau *= 2
        T = int( ( 8*tau*10)**2 )
        TI_trace = np.append(TI_trace, T)
        g0 = func( Gen(T + start -len(g)), start)  # To the previous sample is added what is missing to meet the current T
        g = np.concatenate((g,g0), axis=0) # g doesn't change its matricial shape with one column
        iat = IAT(g)[0,0]
        CV = np.sqrt( iat * np.var(g) ) / mu_B(g) # CV's preliminar estimation 
        CVI_trace = np.append(CVI_trace, CV)
    
    if (tau > 4):
        print("The procedure stopped because the sample of interest has an excessive dispersion.")
        exit() # with this function the procedure stops
        
    ### refinement stage ###
    r = 0 # counter precision
    CV_trace = np.array([]) # this object will save the record of the CV's estimations
    T_trace = np.array([]) # this object will save the record of the T*'s estimations
    T = int( (8*CV*10)**2 ) # CV saves the latest estimation with Bland's method
    T_trace = np.append(T_trace, T)
    if (len(g) < T):
        g0 = func( Gen(T + start -len(g)), start)
        g = np.concatenate((g,g0), axis=0) 
        iat = IAT(g)[0,0]
    CV = np.sqrt( iat * np.var(g) ) / np.mean(g) # Cv with Traditional estimation of mu
    CV_trace = np.append(CV_trace, CV)
    r += 1
    T = int( ( (8*CV)**2 ) * ( 10**(2*r) ) ) 
    T_trace = np.append(T_trace, T)
    
    while (r < p):
        if (len(g) < T):
            g0 = func( Gen(T + start -len(g)), start)
            g = np.concatenate((g,g0), axis=0) 
            iat = IAT(g)[0,0]
        CV = np.sqrt( iat * np.var(g) ) / np.mean(g) # Cv with Traditional estimation of mu
        CV_trace = np.append(CV_trace, CV)
        r += 1
        T = int( ( (8*CV)**2 ) * ( 10**(2*r) ) ) 
        T_trace = np.append(T_trace, T)
        
    if (len(g) < T): # Finally, it is guaranteed that the sample has T* size
            g0 = func( Gen(T + start -len(g)), start)
            g = np.concatenate((g,g0), axis=0) 
            iat = IAT(g)[0,0]
    
    SciNot = SigDigitsMatissaEsp( np.mean(g), p ) # Calculated the matissa of the mean of the sample with presiccion of p sig_digits.
    
    return T , g[:,0] , iat, CVI_trace, TI_trace, CV_trace, T_trace, SciNot  #g[:,0] return a vector
        
def p_cal_MCMC (g_X):
    """
    This function allows to calculate the precision 
    that is guaranteed with a certain dependent sample generated with MCMC 
    from the functional of interest.
    g_X : This is a dependent sample of the functional of interest, in stacionary stage 
    for calcute the IAT, g_X must have matricial shape with one column 
    """
    Tr = len(g_X) # real sample size
    ### initial stage ###            
    tau = 1/4
    T = int( (8*tau*10)**2 )
    if (T > Tr):
        return "With this sample, even zero precision cannot be guaranteed"    
    iat = IAT(g_X[:T,])
    CV = np.sqrt( iat * np.var(g_X[:T,]) ) / mu_B(g_X[:T,]) # CV's preliminar estimation 
    
    while (CV > tau and tau <= 4):
        tau *= 2
        T = int( (8*tau*10)**2 )
        if (T > Tr):
            return "With this sample, even zero precision cannot be guaranteed"    
        iat = IAT(g_X[:T,])
        CV = np.sqrt( iat * np.var(g_X[:T,]) ) / mu_B(g_X[:T,])
    
    if (tau > 4):
        print("The procedure stopped because the sample of interest has an excessive dispersion.")
        exit() # with this function the procedure stops
    
    ### refinement stage ###
    p = 0        
    T = int( (8*CV*10)**2 ) # CV saves the latest estimation with Bland's method
    if (T > Tr):
        return "With this sample, even zero precision cannot be guaranteed"    
    p += 1
    iat = IAT(g_X[:T,])
    CV = np.sqrt( iat * np.var(g_X[:T,]) ) / np.mean(g_X[:T,])
    T = int( ( (8*CV)**2 ) * ( 10**(2*p) ) ) 
    if (T > Tr):
        return print('This sample guarantees precision, p =', p - 1)
    
    while (Tr > T):
        iat = IAT(g_X[:T,])
        CV = np.sqrt( iat * np.var(g_X[:T,]) ) / np.mean(g_X[:T,])
        p += 1
        T = int( ( (8*CV)**2 ) * ( 10**(2*p) ) ) 
    
    SciNot = SigDigitsMatissaEsp( np.mean(g_X), p-1 ) # Calculated the matissa of the mean of the sample with presiccion of p sig_digits.
    
    return print('\nThis sample guarantees precision, p =', p - 1 , \
                 '\n\nThe matissa and the exponent of $\mu$ are:', SciNot )
