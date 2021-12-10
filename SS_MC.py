"""
Criteria for number of runs in stochastic simulations - Case MC simulations
August 2021
Created by jdmolina
"""

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from sys import exit

############################################################
############### Auxiliar Functions  ########################
############################################################

def CV_Bland (X):
    """
    This function calculate a preliminar estimation of the variation coefficient, 
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
    ## Sample variance's estimation
    # Estimation of the sum of the xi^2
    # Lower boundary for the sum of the xi^2
    IS = ( (n+3)*(a**2 + q1**2 + m**2 + q3**2) + 8*b**2 + (n-5)*(a*q1 + q1*m + m*q3 + q3*b) )/8
    # Upper boundary for the sum of the xi^2
    SS = ( 8*a**2 + (n+3)*(q1**2 + m**2 + q3**2 + b**2) + (n-5)*(a*q1 + q1*m + m*q3 + q3*b) )/8
    SX2 = (IS + SS)/2
    S2 = (SX2 - n*X_bar**2)/(n-1)
    ## Estimation of the variation coefficient
    CV = np.sqrt(S2) / X_bar    
    return CV

def SigDigitsMatissaEsp( x, sig_digits=4):
    """
    x is a real number.
    Return the matissa m as a float, 1 <= m < 10 and exponent q as an int.
    """  
    tmp = np.format_float_scientific( x, precision=sig_digits-1, trim='0', exp_digits=3)
    return float(tmp[:-5]), int(tmp[-4:])


############################################################
############### Principal Functions  #######################
############################################################

def T_Est(p, Gen, func):
    """
    This function allows to estimate T^*, for a p given precision, 
    when the mechanism to generate values of the sample 
    and to calculate the functional are known.
    p : determines the precision in the mantissa's estimation
    Gen : function that allows to generate values of the sample of interest, it must return an array of dim T X m (T is the number of simulations and m the dimension of the variable) 
    func : function that calculates the functional of the sample, must return an array of dimension T 
    """
    ### initial stage ###            
    CVI_trace = np.array([]) # this object will save the record of the initial CV's estimations
    TI_trace = np.array([]) # this object will save the record of the initial T*'s estimations
    tau = 1/4
    T = int( (8*tau*10)**2 )
    TI_trace = np.append(TI_trace, T)
    g = func( Gen(T) ) # This object will save the values of the functional of interest
    CV = CV_Bland(g) # CV's preliminar estimation 
    CVI_trace = np.append(CVI_trace, CV)
    
    while (CV > tau and tau <= 4):
        tau *= 2
        T = int( (8*tau*10)**2 )
        TI_trace = np.append(TI_trace, T)
        g0 = func( Gen( int( T - len(g) ) ) ) # To the previous sample will be added what is missing to meet the current T
        g = np.append(g , g0)
        CV = CV_Bland(g) # CV's preliminar estimation 
        CVI_trace = np.append(CVI_trace, CV)
    
    if (tau > 4):
        print("Warning: The procedure stopped because the sample of interest has an excessive dispersion.")
        exit() # with this function the procedure stops
        
    ### refinement stage ###
    r = 0 # precision counter
    CV_trace = np.array([]) # this object will save the record of the CV's estimations
    T_trace = np.array([]) # this object will save the record of the T*'s estimations
    T = int( (8*CV*10)**2 ) # CV saves the latest estimation with Bland's method
    T_trace = np.append(T_trace, T)
    if (len(g) < T):
        g0 = func( Gen( int( T - len(g) ) ) ) # To the previous sample is added what is missing to meet the current T
        g = np.append(g , g0)
    CV = np.std(g) / np.mean(g) # Traditional estimation of the CV
    CV_trace = np.append(CV_trace, CV)
    r += 1
    T = int( ( (8*CV)**2 ) * ( 10**(2*r) ) ) 
    T_trace = np.append(T_trace, T)
    
    while (r < p):
        if (len(g) < T):
            g0 = func( Gen( int( T - len(g) ) ) ) 
            g = np.append(g , g0)
        CV = np.std(g) / np.mean(g) 
        CV_trace = np.append(CV_trace, CV)
        r += 1
        T = int( ( (8*CV)**2 ) * ( 10**(2*r) ) ) 
        T_trace = np.append(T_trace, T)
        
    if (len(g) < T): # Finally, it is guaranteed that the sample has T* size
            g0 = func( Gen( int( T - len(g) ) ) ) 
            g = np.append(g , g0)
    
    SciNot = SigDigitsMatissaEsp( np.mean(g), p ) # Calculated the matissa of the mean of the sample with presiccion of p sig_digits.
    
    return T , g , CVI_trace, TI_trace, CV_trace, T_trace, SciNot 
        
def p_cal (g_X):
    """
    This function allows to calculate the precision in the mantissa's estimation  
    that is guaranteed with a certain independent sample 
    from the functional of interest.
    g_X : This is a independent sample of the functional of interest
    """
    Tr = len(g_X) # true sample size
    ### initial stage ###            
    tau = 1/4
    T = int( (8*tau*10)**2 )
    if (T > Tr):
        return "With this sample, even zero precision cannot be guaranteed"    
    CV = CV_Bland( g_X[:T] ) # CV's preliminar estimation 
    
    while (CV > tau and tau <= 4):
        tau *= 2
        T = int( (8*tau*10)**2 )
        if (T > Tr):
            return "With this sample, even zero precision cannot be guaranteed"    
        CV = CV_Bland( g_X[:T] )
    
    if (tau > 4):
        print("The procedure stopped because the sample of interest has an excessive dispersion.")
        exit() # with this function the procedure stops
    
    ### refinement stage ###
    p = 0        
    T = int( ( (8*CV*10)**2 ) ) # CV saves the latest estimation with Bland's method
    if (T > Tr):
        return "With this sample, even zero precision cannot be guaranteed"    
    p += 1
    CV = np.std(g_X[:T]) / np.mean(g_X[:T]) 
    T = int( ( (8*CV)**2 ) * ( 10**(2*p) ) )  
    if (T > Tr):
        return print('This sample guarantees precision, p =', p - 1)
    
    while (Tr > T):
        CV = np.std(g_X[:T]) / np.mean(g_X[:T]) 
        p += 1
        T = int( ( (8*CV)**2 ) * ( 10**(2*p) ) )  
    
    SciNot = SigDigitsMatissaEsp( np.mean(g_X), p-1 ) # Calculated the matissa of the mean of the sample with presiccion of p sig_digits.
    
    return print('\nThis sample guarantees precision, p =', p - 1 , \
                 '\n\nThe matissa and the exponent of $\mu$ are:', SciNot )
    