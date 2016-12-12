# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 13:36:12 2016

@author: De Bortoli
"""

import numpy as np
from numpy import random

def generative_fun(pi_l,A_l,C_m,W_l,T):
    M=len(pi_l)
    D=len(C_m)
    
    X_m=np.zeros((M,T+1))
    Y_m=np.zeros((D,T+1))
    
    for m in range(M):
        r=random.random()
        cumpim_v=np.cumsum(pi_l[m])
        ind_v=(r<cumpim_v)*1
        indm=np.argmax(ind_v)
        X_m[m,0]=indm
    
    for t in range(1, T+1):
        Xoldt_v=X_m[:,t-1]
        for m in np.arange(M):
            Am_m=A_l[m]
            Xoldtm=Xoldt_v[m]
            prob_v=Am_m[Xoldtm,:]
            
            r=random.random()
            cumprob_v=np.cumsum(prob_v)
            ind_v=(r<cumprob_v)*1
            indtm=np.argmax(ind_v)
            X_m[m,t]=indtm
            
    for t in range(T+1):
        mut_v=0
        for m in np.arange(M):
            mut_v+=W_l[m][:,X_m[m,t]]
            
        Yt_v=np.random.multivariate_normal(mut_v,C_m)
        Y_m[:,t]=Yt_v
        
    return [X_m,Y_m]