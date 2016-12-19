# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 13:04:38 2016

@author: De Bortoli
"""

import numpy as np
from numpy import *
det=np.linalg.det
inv=np.linalg.inv

def logprobchain(ind1,ind2,logA_l):
    """
    Compute the logarithm of the probability of jumping from the M states ind1
    to the M states ind2.
    
    Input arguments:
        ind1 -- a tuple of size M containing the initial M states
        ind2 -- a tuple of size M containing the final M states
        logA_l -- a list of length M where each element corresponds to the
                  logarithm of the transition matrix of the m-th Markov chain.
                  
    Output arguments:
        p -- a float correspond to the logarithm of the desired probability.
    """
    p = 0
    M = len(logA_l)
    for m in np.arange(M):
        p += logA_l[m][ind1[m],ind2[m]]
        
    return p
    
def logprobobs(t,ind,C_m,W_l,Y_m):
    """
    Compute the logarithm of the probability of observing a state at time t
    knowing the hidden variables.
    
    Input arguments:
        t -- an integer between 0 and T corresponding to the time.
        ind -- a tuple of size M containing the M hidden variables.
        C_m -- the covariance matrix.
        W_l -- a list of length M where each element corresponds to the mean
               contribution of the M-th Markov chain.
        Y_m -- a matrix of size Dx(T+1) where each column corresponds to an
               observation at time t.
               
    Output arguments:
        p -- a float correspond to the logarithm of the desired probability.
    """
    M = len(W_l)
    mu_v = 0
    
    for m in np.arange(M):
        mu_v += W_l[m][:,ind[m]]
    
    p = (-1/2*log(2*pi) - 1/2*log(det(C_m)) - 
        1/2*dot(Y_m[:,t]-mu_v, dot(inv(C_m), Y_m[:,t]-mu_v)))
    
    return p
    