# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 22:41:36 2016

@author: De Bortoli
"""
import numpy as np
from sumprodlog import sumprodlog
from numpy import matlib, linalg
from exact_inference import exact_inference, exact_inference_E
from gibbs_sampler import gibbs_sampler, gibbs_sampler_E

def EM(pi0_l, A0_l, C0_m, W0_l, Y_m, N, method=1, Ns=10):
    """
    Apply the EM algorithm to a dataset with given E step method.
    
    Input arguments:
        pi0_l -- a list of length M where each element corresponds to the 
                 initial distribution of the m-th Markov chain.
        A0_l -- a list of length M where each element corresponds to the 
                transition matrix of the m-th Markov chain.
        C0_m -- the covariance matrix.
        W0_l -- a list of length M where each element corresponds to the mean
                contribution of the M-th Markov chain.
        Y_m -- a matrix of size Dx(T+1) where each column corresponds to an
               observation at time t.
        N -- an integer equal to the number of iterations in the EM algorithm.
        method -- an integer setting the method used during the E step:
                    1 -- exact inference
                    2 -- Gibbs sampling
                    3 -- mean fields
                    4 -- structured mean fields
                    (default value method=1)
    
    Output arguments:
        pi_l -- a list of length M where each element corresponds to the 
                 initial distribution of the m-th Markov chain.
        A_l -- a list of length M where each element corresponds to the 
                transition matrix of the m-th Markov chain.
        C_m -- the covariance matrix.
        W_l -- a list of length M where each element corresponds to the mean
                contribution of the M-th Markov chain.
        L_l -- a list of size N+1 where each element is a float, the
               approximate log-likelihood at iteration n
    """
    #Defining the constants
    M = len(pi0_l)
    K=len(pi0_l[0])
    
    #Initializing the parameters
    pi_l=pi0_l
    A_l=A0_l
    C_m=C0_m
    W_l=W0_l
    
    #Initializing thee log-likelihood
    L_l=[]
    
    for n in range(N):
        #Initializing the logarithm of the parameters
        logpi_l=log(pi_l)
        logA_l=log(A_l)
        
        #E step
        if method == 1:
            [alpha_l, beta_l, gamma_l] = exact_inference(logpi_l, logA_l, \
            C_m, W_l, Y_m)
            [E_l, EmixM_l, EmixT_l] = exact_inference_E(alpha_l, beta_l, \
            gamma_l, logA_l, C_m, W_l, Y_m)
        elif method == 2:
            [S_l] = gibbs_sampler(logpi_l, logA_l, C_m, W_l, Y_m, Ns)
            [E_l, EmixM_l, EmixT_l] = gibbs_sampler_E(S_l, K)
        
        #M step
        [pi_l, A_l, C_m, W_l] = Mstep(E_l, EmixM_l, EmixT_l, Y_m)
        
        #Correction step
        for m in range(M):
            pi_l[m] += 1e-308
            A_l[m] += 1e-308
        
        L_l.append(loglikelihood(pi_l, A_l, C_m, W_l, Y_m, E_l, EmixM_l, \
                                 EmixT_l))
    
    return [pi_l, A_l, C_m, W_l, L_l]

def Mstep(E_l, EmixM_l, EmixT_l, Y_m):
    """
    Compute the M step of the EM algorithm using an exact or an estimated E 
    step.
    
    Input arguments:
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
        Y_m -- a matrix of size Dx(T+1) where each column corresponds to an
               observation at time t.
                    
    Output arguments:
        pi_l -- a list of length M where each element corresponds to the 
                initial distribution of the m-th Markov chain.
        A_l -- a list of length M where each element corresponds to the 
               transition matrix of the m-th Markov chain.
        C_m -- the covariance matrix.
        W_l -- a list of length M where each element corresponds to the mean
               contribution of the M-th Markov chain.
    """
    
    #Defining the constants
    T = len(Y_m[0,:])
    T = T - 1
    D = len(Y_m[:,0])
    M = len(E_l[0][:,0])
    K = len(E_l[0][0,:])
    
    #Filling pi_l
    pi_l = []
    for m in range(M):
        pi_l.append(E_l[0][m,:])
        
    #Filling A_l
    A_l = []
    for m in range(M):
        A_m = np.zeros((K,K))
        S_v = np.zeros((K,1))
        for t in range(1,T+1):
            A_m += EmixT_l[t][m,:,:]
            S_v += np.transpose(np.array([E_l[t-1][m,:]]))
        S_m = np.matlib.repmat(S_v, 1, K)
        A_l.append((A_m+1e-308) / (S_m+1e-308))
        
    #Creating appropriate arrays
    num_m=np.zeros((D,K*M))
    denom_m=np.zeros((K*M,K*M))
    for t in range(T+1):
        Y_v = np.transpose(np.array([Y_m[:,t]]))
        El_v = np.reshape(E_l[t], (1,K*M))
        num_m += np.dot(Y_v, El_v)
        
        for i in range(K*M):
            m_1 = i // K
            k_1 = i - m_1*K
            for j in range(K*M):
                m_2 = j // K
                k_2 = j - m_2*K
                denom_m[i,j] += EmixM_l[t][m_1][m_2][k_1][k_2]
    
    denom_m = np.linalg.pinv(denom_m)
    
    #Computing W and filling W_l
    W = np.dot(num_m,denom_m)
    W_l = []
    
    for m in range(M):
        W_l.append(W[:,m*K:(m+1)*K])
        
    #Computing C
    C_m=np.zeros((D,D))
    
    for t in range(T+1):
        Y_v = np.transpose(np.array([Y_m[:,t]]))
        C_m += np.dot(Y_v, np.transpose(Y_v))
        
        Y_v = np.transpose(np.array([Y_m[:,t]]))
        El_v = np.reshape(E_l[t], (1,K*M))
        C_m += -np.dot(W, np.transpose(np.dot(Y_v, El_v)))
        
    C_m = C_m/(T+1)
    
    return [pi_l, A_l, C_m, W_l]