# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 10:11:49 2016

@author: De Bortoli
"""
import numpy as np
from logprob import logprobobs

def loglikelihood(pi_l, A_l, C_m, W_l, Y_m, E_l, EmixM_l, EmixT_l):
    """
    Compute the approximate log-likelihood of the model for given parameters.
    
    Input arguments:
        pi_l -- a list of length M where each element corresponds to the 
                 initial distribution of the m-th Markov chain.
        A_l -- a list of length M where each element corresponds to the 
                transition matrix of the m-th Markov chain.
        C_m -- the covariance matrix.
        W_l -- a list of length M where each element corresponds to the mean
                contribution of the M-th Markov chain.
        Y_m -- a matrix of size Dx(T+1) where each column corresponds to an
               observation at time t.
        E_l -- a list of size (T+1) where each element corresponds to a matrix
               of size MxK containing the expectation of getting the k-th value
               at time t and chain m.
        EmixM_l -- a list of size (T+1) where each element corresponds to a
                   matrix of size MxMxKxK tensor containing the expectation of
                   getting the k1-th value at time t and chain m1 and getting
                   the k2-th value at time t and chain m2.
        EmixT_l -- a list of size (T+1) where each element corresponds to a 
                   matrix of size MxKxK tensor containing the expectation of 
                   getting the k1-th value at time t-1 and chain m and getting 
                   the k2-th value at time t and chain m.
               
    Output arguments:
        L -- a float corresponding to the approximate log-likelihood.
    """
    #Defining the constants
    M = len(pi_l)
    T = len(Y_m[0,:])
    T = T - 1
    
    #Initializing the logarithm of the parameters and the loglikelihood
    logpi_l = log(pi_l)
    logA_l = log(A_l)
    L = 0
    
    #Computing L for the 0 states
    for m in range(M):
        L += sum(logpi_l[m]*E_l[0][m,:])
        
    #Adding the dependencies between the hidden variables
    for t in range(1,T+1):
        for m in range(M):
            L += sum(logA_l[m]*EmixT_l[t][m,:,:])
            
    #Adding the observations
    for t in range(T+1):
        Y_v = np.transpose(np.array([Y_m[:,t]]))
        L += -1/2 * np.dot(np.dot(np.transpose(Y_v), np.linalg.inv(C_m)), \
             Y_v)[0,0]
        for m in range(M):
            L += np.dot(np.transpose(Y_v), np.dot(np.linalg.inv(C_m), \
                 np.dot(W_l[m], np.transpose(np.array([E_l[t][m,:]])))))[0,0]
            for n in range(M):
                L += -1/2*np.trace(np.dot(np.transpose(W_l[m]), \
                     np.dot(np.linalg.inv(C_m), np.dot(W_l[n], \
                     EmixM_l[t][n,m,:,:]))))
    
    return [L]