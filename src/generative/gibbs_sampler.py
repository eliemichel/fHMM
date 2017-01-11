# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:21:15 2016

@author: De Bortoli
"""

from logprob import logprobobs

import random
import numpy as np
from scipy import misc as sp_misc


def gibbs_sampler(logpi_l, logA_l, C_m, W_l, Y_m, Ns):
    """
    Sample using Gibbs sampling

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
        Ns -- an integer corresponding to the number of samples

    Output arguments:
        S_l -- a list of size Ns+1 where each element is a matrix of size
               Mx(T+1) where each column corresponds to the hidden variables at
               time t.
    """

    # constants
    M = len(logpi_l)
    K = len(logpi_l[0])
    T = len(Y_m[0, :])
    T = T - 1

    # initialization
    S_l = []
    for n in range(Ns+1):
        S_l.append(np.zeros((M, T+1), dtype=np.int))

    # uniform random filling of the first sample
    for t in range(T+1):
        for m in range(M):
            S_l[0][m, t] = np.random.randint(K)

    # filling recursion
    for n in range(1, Ns+1):
        # special case: states at time 0
        S0int_v = np.copy(S_l[n-1][:, 0])

        for m in range(M):
            logp_v = logpi_l[m] + logA_l[m][:, S_l[n-1][m, 1]]
            logprobobs_v = np.zeros((K,))

            for k in range(K):
                ind = S0int_v[0:m].tolist() + [k] + S0int_v[(m+1):M].tolist()
                logprobobs_v[k] = logprobobs(0, tuple(ind), C_m, W_l, Y_m)

            logp_v += logprobobs_v
            logp_v += -sp_misc.logsumexp(logp_v)
            cump_v = np.cumsum(np.exp(logp_v))

            r = random.random()
            ind_v = int(r < cump_v)
            indm = np.argmax(ind_v)
            S0int_v[m] = indm

        S_l[n][:, 0] = S0int_v

        # standard case: states at times 1...T-1
        for t in range(1, T):
            Stint_v = np.copy(S_l[n-1][:, t])

            for m in range(M):
                logp_v = (logA_l[m][:, S_l[n-1][m, t+1]] +
                          logA_l[m][S_l[n][m, t-1], :])
                logprobobs_v = np.zeros((K,))

                for k in range(K):
                    ind = (Stint_v[0:m].tolist() + [k] +
                           Stint_v[(m+1):M].tolist())
                    logprobobs_v[k] = logprobobs(t, tuple(ind), C_m, W_l, Y_m)

                logp_v += logprobobs_v
                logp_v += -sp_misc.logsumexp(logp_v)
                cump_v = np.cumsum(np.exp(logp_v))

                r = random.random()
                ind_v = int(r < cump_v)
                indm = np.argmax(ind_v)
                Stint_v[m] = indm

            S_l[n][:, t] = Stint_v

        # special case: states at time T
        STint_v = np.copy(S_l[n-1][:, T])

        for m in range(M):
            logp_v = logA_l[m][S_l[n][m, T-1], :]
            logprobobs_v = np.zeros((K,))

            for k in range(K):
                ind = STint_v[0:m].tolist() + [k] + STint_v[(m+1):M].tolist()
                logprobobs_v[k] = logprobobs(T, tuple(ind), C_m, W_l, Y_m)

            logp_v += logprobobs_v
            logp_v += -sp_misc.logsumexp(logp_v)
            cump_v = np.cumsum(np.exp(logp_v))

            r = random.random()
            ind_v = int(r < cump_v)
            indm = np.argmax(ind_v)
            STint_v[m] = indm

        S_l[n][:, T] = STint_v

    return [S_l]


def gibbs_sampler_E(S_l, K):
    """
    E-step using Gibbs sampling.

    Input arguments:
        S_l -- a list of size Ns+1 where each element is a matrix of size
               Mx(T+1) where each column corresponds to the hidden variables at
               time t.
        K -- an integer corresponding to the size of the finite set where the
             hidden variables take their values.

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

    # constants
    Ns = len(S_l) - 1
    M = len(S_l[0][:, 0])
    T = len(S_l[0][0, :]) - 1

    # initialization
    E_l = []
    EmixM_l = []
    EmixT_l = []

    for t in range(T+1):
        E_l.append(np.zeros((M, K)))
        EmixM_l.append(np.zeros((M, M, K, K)))
        EmixT_l.append(np.zeros((M, K, K)))

    # fill E_l
    for n in range(1, Ns+1):
        for t in range(T+1):
            for m in range(M):
                E_l[t][m, S_l[n][m, t]] += 1
    E_l[:] = [M/Ns for M in E_l]

    # fill EmixM_l
    for n in range(1, Ns+1):
        for t in range(T+1):
            for m1 in range(M):
                for m2 in range(M):
                    EmixM_l[t][m1, m2, S_l[n][m1, t], S_l[n][m2, t]] += 1
    EmixM_l[:] = [M/Ns for M in EmixM_l]

    # fill EmixT_l
    for n in range(1, Ns+1):
        for t in range(1, T+1):
            for m in range(M):
                EmixT_l[t][m, S_l[n][m, t-1], S_l[n][m, t]] += 1
    EmixT_l[:] = [M/Ns for M in EmixT_l]

    return [E_l, EmixM_l, EmixT_l]
