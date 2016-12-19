# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 19:40:51 2016

@author: De Bortoli
"""
import numpy as np
import itertools
from numpy import *
from logprob import logprobobs,logprobchain
from sumprodlog import sumprodlog
from extract_mat import extract_mat

def exact_inference(logpi_l, logA_l, C_m, W_l, Y_m):
    """
    Compute alpha, beta and gamma according to [1].
    
    Input arguments:
        logpi_l -- a list of length M where each element corresponds to the
                   logarithm of the initial distribution of the m-th Markov
                   chain.
        logA_l -- a list of length M where each element corresponds to the
                  logarithm of the transition matrix of the m-th Markov chain.
        C_m -- the covariance matrix.
        W_l -- a list of length M where each element corresponds to the mean
               contribution of the M-th Markov chain.
        Y_m -- a matrix of size Dx(T+1) where each column corresponds to an
               observation at time t.
        
    Output arguments:
        alpha_l -- a list of size (T+1) where each element corresponds to a
                   tensor of size Kx..xK (M times) containing the logarithm 
                   of the alpha recursion.
        beta_l -- a list of size (T+1) where each element corresponds to a
                  tensor of size Kx..xK (M times) containing the logarithm of 
                  the beta recursion. 
        gamma_l -- a list of size (T+1) where each element corresponds to a
                   tensor of size Kx..xK (M times) containing the logarithm of 
                   the gamma recursion.
                   
    References:
        [1]: Factorial Hidden Markov Models, Ghahramni, Z. & Jordan, M.I.
             Machine Learning (1997)
    """
    
    #Defining the constants
    M = len(logpi_l)
    K = len(logpi_l[0])
    T = len(Y_m[0,:])
    T = T - 1
    
    #Initializing the alpha & beta lists
    alpha_l = []
    beta_l = []
    
    for t in range(T+1):
        alpha_l.append(np.ndarray(tuple([K]*M))*0)
        beta_l.append(np.ndarray(tuple([K]*M))*0)
        
    #Filling alpha_l[0] (initialization step)
    for ind in itertools.product(*([range(K)]*M)):
        prob = 0
        for m in range(M):
            prob += logpi_l[m][ind[m]]
            
        prob += logprobobs(0,ind,C_m,W_l,Y_m)
        alpha_l[0][ind]=prob
    #Remark: beta_l[T] already initialized
        
    #Recursion step
    for t in range(1,T+1):
        alphatm_m = np.copy(alpha_l[t-1])
        betatm_m = np.copy(beta_l[T-t+1])
        
        #Getting beta(T-t)(M) from beta(T-t+1)
        for ind in itertools.product(*([range(K)]*M)):
            betatm_m[ind] += logprobobs(T-t+1,ind,C_m,W_l,Y_m)
        #Remark: we already have alpha(t+1)(M) from alpha(t)
        
        #M backward recursion step
        for m in range(M-1,-1,-1):
            alphatmold_m = alphatm_m
            betatmold_m = betatm_m
            
            for ind in itertools.product(*([range(K)]*M)):
                [extractalphatmold_v] = extract_mat(ind, alphatmold_m, m)
                [extractbetatmold_v] = extract_mat(ind, betatmold_m, m)
                alphatm_m[ind] = (sumprodlog(logA_l[m][:,ind[m]] + 
                                 extractalphatmold_v))
                betatm_m[ind] = (sumprodlog(logA_l[m][ind[m],:] + 
                                extractbetatmold_v))
                
        #Getting alpha(t+1) from alpha(t+1)(0)    
        for ind in itertools.product(*([range(K)]*M)):
            alphatm_m[ind] += logprobobs(t,ind,C_m,W_l,Y_m)
        #Remark: we already have beta(T-t) from beta(T-t)(0)
        
        alpha_l[t] = alphatm_m
        beta_l[T-t] = betatm_m
    
    #Computing gamma  
    gamma_l = []
    
    for t in range(T+1):
        gamma_l.append(alpha_l[t] + beta_l[t] - \
                        sumprodlog(alpha_l[t]+beta_l[t]))
        
    
    return [alpha_l,beta_l,gamma_l]
    
def exact_inference_E(alpha_l,beta_l,gamma_l,logA_l,C_m,W_l,Y_m):
    """
    E-step using exact inference.
    
    Input arguments:
        alpha_l -- a list of size (T+1) where each element corresponds to a
                   tensor of size Kx..xK (M times) containing the logarithm 
                   of the alpha recursion.
        beta_l -- a list of size (T+1) where each element corresponds to a
                  tensor of size Kx..xK (M times) containing the logarithm of 
                  the beta recursion. 
        gamma_l -- a list of size (T+1) where each element corresponds to a
                   tensor of size Kx..xK (M times) containing the logarithm of 
                   the gamma recursion.
        logA_l -- a list of length M where each element corresponds to the
                  logarithm of the transition matrix of the m-th Markov chain.
        C_m -- the covariance matrix.
        W_l -- a list of length M where each element corresponds to the mean
               contribution of the M-th Markov chain.
        Y_m -- a matrix of size Dx(T+1) where each column corresponds to an
               observation at time t.
             
    Output arguments:
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
                   
    Remark:
        One can note that EmixT_l is not well-defined for t=0. In that case we
        state EmixT_l=np.ndarray((M,K,K))*0
    """
    
    #Defining the constants
    T = len(Y_m[0,:]) - 1
    M = alpha_l[0].ndim
    K = len(alpha_l[0])
    
    #Initializing the lists
    E_l = []
    EmixM_l = []
    EmixT_l = []
    
    for t in range(T+1):
        E_l.append(np.ndarray((M,K))*0)
        EmixM_l.append(np.ndarray((M,M,K,K))*0)
        EmixT_l.append(np.ndarray((M,K,K))*0)
    
    #Initializing the intermediate lists
    Eint_l = ([[[[-1e308] for k in range(K)] for m in range(M)] 
             for t in range(T+1)])
    EmixMint_l = ([[[[[[-1e308] for k1 in range(K)] for k2 in range(K)] 
                 for m1 in range(M)] for m2 in range(M)] for t in range (T+1)])
    EmixTint_l = ([[[[[-1e308] for k1 in range(K)] for k2 in range(K)] 
                 for m in range(M)] for t in range(T+1)])
    Z_l = [[] for t in range(T+1)]
    #Remark: -1e308 is used to produce a zero when applying the exponential
    
    #Filling Eint_l & EmixMint_l
    for t in range(T+1):
        for ind in itertools.product(*([range(K)]*M)):
            for m in range(M):
                Eint_l[t][m][ind[m]] += [gamma_l[t][ind]]
                for n in range(M):
                    EmixMint_l[t][m][n][ind[m]][ind[n]] += [gamma_l[t][ind]]
    
    #Filling EmixTint_l
    for t in range(1,T+1):
        for ind1 in itertools.product(*([range(K)]*M)):
            for ind2 in itertools.product(*([range(K)]*M)):
                p = (alpha_l[t-1][ind1] + beta_l[t][ind2] +
                logprobobs(t, ind2, C_m, W_l, Y_m) + 
                logprobchain(ind1, ind2, logA_l))
                Z_l[t] += [p]
                for m in range(M):
                    EmixTint_l[t][m][ind1[m]][ind2[m]] += [p]
             
    #Filling E_l
    for t in range(T+1):
        for m in range(M):
            for k in range(K):
                E_l[t][m,k] = sumprodlog(np.array(Eint_l[t][m][k]))
        E_l[t] = exp(E_l[t])
    
    #Filling EmixM_l         
    for t in range(T+1):
        for m1 in range(M):
            for m2 in range(M):
                for k1 in range(K):
                    for k2 in range(K):
                        EmixM_l[t][m1,m2,k1,k2] = \
                        sumprodlog(np.array(EmixMint_l[t][m1][m2][k1][k2]))
        EmixM_l[t] = exp(EmixM_l[t])
      
    #Filling EmixT_l
    for t in range(1,T+1):
        Zt = sumprodlog(np.array(Z_l[t]))
        for m in range(M):
            for k1 in range(K):
                for k2 in range(K):
                    EmixT_l[t][m,k1,k2] = \
                    sumprodlog(np.array(EmixTint_l[t][m][k1][k2])) - Zt
        EmixT_l[t] = exp(EmixT_l[t])
                    
    
    return [E_l,EmixM_l,EmixT_l]