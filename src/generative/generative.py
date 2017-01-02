# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 13:36:12 2016

@author: De Bortoli
"""

import numpy as np
from numpy import random

def generative(pi_l, A_l, C_m, W_l, T):
    """
    Generate hidden variables and observation according to the factorial hidden
    Markov model.
    
    Input arguments:
        pi_l -- a list of length M where each element corresponds to the
                initial distribution of the m-th Markov chain.
        A_l -- a list of length M where each element corresponds transition 
               matrix of the m-th Markov chain.
        C_m -- the covariance matrix.
        W_l -- a list of length M where each element corresponds to the mean
               contribution of the M-th Markov chain.
        T -- an integer corresponding to the final time.
        
    Output arguments:
        S_m -- a matrix of size Mx(T+1) where each element corresponds to the 
               value of the m-th hidden variable at time t.
        Y_m -- a matrix of size Dx(T+1) where each column corresponds to an
               observation at time t.
    """
    
    #Defining the constants
    M = len(pi_l)
    D = len(C_m)
    
    #Initializing the matrices
    S_m = np.zeros((M,T+1))
    Y_m = np.zeros((D,T+1))
    
    #Filling the states at time 0
    for m in range(M):
        r = random.random()
        cumpim_v = np.cumsum(pi_l[m])
        ind_v = (r<cumpim_v)*1
        indm = np.argmax(ind_v)
        S_m[m,0] = indm
    
    #Recursion for the states at time 1,...,T
    for t in range(1,T+1):
        Soldt_v = np.copy(S_m[:,t-1])
        for m in np.arange(M):
            Am_m = A_l[m]
            Soldtm_v = Soldt_v[m]
            prob_v = Am_m[np.int(Soldtm_v),:]
            
            r = random.random()
            cumprob_v = np.cumsum(prob_v)
            ind_v = (r<cumprob_v)*1
            indtm = np.argmax(ind_v)
            S_m[m,t] = indtm
    
    #Building the observation for each time        
    for t in range(T+1):
        mut_v = 0
        for m in np.arange(M):
            mut_v += W_l[m][:,np.int(S_m[m,t])]
            
        Yt_v = np.random.multivariate_normal(mut_v, C_m)
        Y_m[:,t] = Yt_v
        
    return [S_m,Y_m]
    
def randinit(M, K, D):
    """
    Give a random initialization to the EM algorithm.
    
    Input arguments:
        M -- an integer representing the number of Markov chains.
        K -- an integer representing the number of states
        D -- an integer representing the dimension of the output arguments.
        
    Output arguments:
        pi_l -- a list of length M where each element corresponds to the 
                initial distribution of the m-th Markov chain.
        A_l -- a list of length M where each element corresponds to the 
               transition matrix of the m-th Markov chain.
        W_l -- a list of length M where each element corresponds to the mean
               contribution of the M-th Markov chain.
    """
    
    #Filling pi_l
    pi_l = []
    for m in range(M):
        piint_v = np.reshape(random.rand(1,K), (K,))
        pi_l.append(piint_v / sum(piint_v))
    
    #Filling A_l
    A_l = []
    for m in range(M):
        Aint_m = random.rand(K,K)
        for k in range(K):
            s = sum(Aint_m[k,:])
            Aint_m[k,:] = Aint_m[k,:] / s
        A_l.append(Aint_m)
    
    #Filling W_l
    W_l = []
    for m in range(M):
        W_l.append(random.rand(D, K))
    
    return [pi_l, A_l, W_l]