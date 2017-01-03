# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 14:23:39 2017

@author: De Bortoli
"""

import numpy as np

def meanfield(logpi_l, logA_l, C_m, W_l, Y_m, Nit):
    """
    Create the mean field interpretation of the model given the parameters.
    
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
        Nit -- an integer corresponding to the number of iterations in the 
               Gauss-Seidel computation to solve the fixed point problem.
               
    Output arguments:
        theta_l -- a list of size T+1 where each element is a list of size M 
                   where each element is a vector of size K representing the 
                   probability associated with the mean field model at time t 
                   and chain m.
    """
    #Defining the constants
    M = len(logpi_l)
    K = len(logpi_l[0])
    T = len(Y_m[0,:]) - 1
    D = len(Y_m[:,0])
    
    #Initialization of the state probabilities
    theta_l = [[np.zeros((K,1)) for m in range(M)] for t in range(T+1)]
    
    for t in range(T+1):
        for m in range(M):
            thetaint_v = np.random.rand(1, K)
            thetaint_v = thetaint_v / sum(thetaint_v)
            theta_l[t][m] = np.transpose(thetaint_v)
    
    #Recursion using Gauss-Seidel
    for n in range(Nit):
        
        #States at time 0
        Y0_v = np.reshape(Y_m[:,0], (D,1))
        for m in range(M):
            
            #creating residual vector and diagonal matrix
            Yres_v = np.copy(Y0_v)
            diagint_m = (np.dot(np.transpose(W_l[m]), np.dot(np.linalg.inv(C_m),
                                            W_l[m])))
            diag_v = np.reshape(np.diag(diagint_m), (K,1))
            for l in range(M):
                if l != m:
                    Yres_v += -np.dot(W_l[l], theta_l[0][l])
            
            logthetaint0m_v = (np.dot(np.transpose(W_l[m]), 
                              np.dot(np.linalg.inv(C_m), Yres_v)) -1/2*diag_v +
                              np.dot(np.transpose(logA_l[m]), theta_l[1][m]) +
                              np.reshape(logpi_l[m], (K,1)))
            logthetaint0m_v = logthetaint0m_v - sumprodlog(logthetaint0m_v)
            theta_l[0][m] = exp(logthetaint0m_v)
        
        
        #States at time t in  1,..,T-1
        for t in range(1,T):
            Yt_v = np.reshape(Y_m[:,t], (D,1))
            for m in range(M):
                
                #creating residual vector and diagonal matrix
                Yres_v = np.copy(Yt_v)
                diagint_m = ((np.dot(np.transpose(W_l[m]), 
                                    np.dot(np.linalg.inv(C_m), W_l[m]))))
                diag_v = np.reshape(np.diag(diagint_m), (K,1))
                for l in range(M):
                    if l != m:
                        Yres_v += -np.dot(W_l[l], theta_l[t][l])
                    
                logthetainttm_v = (np.dot(np.transpose(W_l[m]), 
                              np.dot(np.linalg.inv(C_m), Yres_v)) -1/2*diag_v +
                              np.dot(np.transpose(logA_l[m]), theta_l[t+1][m]) +
                              np.dot(logA_l[m], theta_l[t-1][m]))
                logthetainttm_v = logthetainttm_v - sumprodlog(logthetainttm_v)
                theta_l[t][m] = exp(logthetainttm_v)
         
        #States at time T
        YT_v = np.reshape(Y_m[:,T], (D,1))
        for m in range(M):
            
            #creating residual vector and diagonal matrix
            Yres_v = np.copy(YT_v)
            diagint_m = (np.dot(np.transpose(W_l[m]), np.dot(np.linalg.inv(C_m),
                                            W_l[m])))
            diag_v = np.reshape(np.diag(diagint_m), (K,1))
            for l in range(M):
                if l != m:
                    Yres_v += -np.dot(W_l[l], theta_l[T][l])
            
            logthetaintTm_v = (np.dot(np.transpose(W_l[m]), 
                              np.dot(np.linalg.inv(C_m), Yres_v)) -1/2*diag_v +
                              np.dot(logA_l[m], theta_l[t-1][m]))
            logthetaintTm_v = logthetaintTm_v - sumprodlog(logthetaintTm_v)
            theta_l[T][m] = exp(logthetaintTm_v)
        
        
    return [theta_l]
    
def meanfield_E(theta_l):
    """
    E-step using mean field.
    
    Input arguments:
        theta_l -- a list of size T+1 where each element is a list of size M 
                   where each element is a vector of size K representing the 
                   probability associated with the mean field model at time t 
                   and chain m.
             
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
    T = len(theta_l) - 1
    M = len(theta_l[0])
    K = len(theta_l[0][0])
    
    #Initializing the lists
    E_l=list()
    EmixM_l=list()
    EmixT_l=list()
    
    for t in range(T+1):
        E_l.insert(t,np.ndarray((M,K))*0)
        EmixM_l.insert(t,np.ndarray((M,M,K,K))*0)
        EmixT_l.insert(t,np.ndarray((M,K,K))*0)
        
    #Filling E_l
    for t in range(T+1):
        for m in range(M):
            for k in range(K):
                E_l[t][m,k] = theta_l[t][m][k]
                
    #Filling EmixM_l
    for t in range(T+1):
        for m1 in range(M):
            for m2 in range(M):
                if m2 != m1:
                    for k1 in range(K):
                        for k2 in range(K):
                            EmixM_l[t][m1,m2,k1,k2] = theta_l[t][m1][k1] * \
                            theta_l[t][m2][k2]
                else:
                    for k in range(K):
                        EmixM_l[t][m1,m1,k,k] = theta_l[t][m1][k]
                        
    #Filling EmixT_l
    for t in range(1,T+1):
        for m in range(M):
            for k1 in range(K):
                for k2 in range(K):
                    EmixT_l[t][m,k1,k2] = theta_l[t-1][m][k1] * theta_l[t][m][k2]
    
    return [E_l,EmixM_l,EmixT_l]