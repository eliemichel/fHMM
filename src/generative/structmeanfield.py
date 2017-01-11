# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 01:40:07 2017

@author: De Bortoli
"""

import scipy.misc as sp_misc
import numpy as np


def structmeanfield(logpi_l, logA_l, C_m, W_l, Y_m, Nit):
    """
    Create the structured mean field interpretation of the model given the
    parameters.

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
        h_l -- a list of size T+1 where each element is a list of size M where
               each element is a vector of size K representing the parameter
               associated with the structured mean field model at time t and
               chain m.
    """
    M = len(logpi_l)
    K = len(logpi_l[0])
    T = len(Y_m[0, :]) - 1
    D = len(Y_m[:, 0])

    h_l = [[np.zeros((K, 1)) for m in range(M)] for t in range(T+1)]

    for t in range(T+1):
        for m in range(M):
            h_l[t][m] = np.random.random((K, 1))

    # recursion step (minimizing the Kullback-Leiber divergence)
    for n in range(Nit):
        # compute the expectations
        [E_l, EmixM_l, EmixT_l] = structmeanfield_E(logpi_l, logA_l, h_l)

        # compute and store the deltas
        delta = [np.zeros((K, 1)) for m in range(M)]
        for m in range(M):
            diagint_m = (np.dot(np.transpose(W_l[m]),
                                np.dot(np.linalg.inv(C_m),
                                       W_l[m])))
            delta[m] = np.reshape(np.diag(diagint_m), (K, 1))

        for t in range(T+1):
            Yt_v = np.reshape(Y_m[:, t], (D, 1))
            for m in range(M):
                Yres_v = np.copy(Yt_v)
                for n in range(M):
                    if n != m:
                        Yres_v += -(np.dot(W_l[n],
                                           np.reshape(E_l[t][n, :], (K, 1))))
                h_l[t][m] = (np.exp(np.dot(
                    np.transpose(W_l[m]),
                    np.dot(np.linalg.inv(C_m), Yres_v)) -
                                    1/2*delta[m]))

    return [h_l]


def structmeanfield_E(logpi_l, logA_l, h_l):
    """
    E-step using structured mean field. This implementation relies on a message
    passing algorithm.

    Input arguments:
        logpi_l -- a list of length M where each element corresponds to the
                   logarithm of the initial distribution of the m-th Markov
                   chain.
        logA_l -- a list of length M where each element corresponds to the
                  logarithm of the transition matrix of the m-th Markov chain.
        h_l -- a list of size T+1 where each element is a list of size M where
               each element is a vector of size K representing the parameter
               associated with the structured mean field model at time t and
               chain m.

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
    logh_l = np.log(h_l)

    T = len(h_l) - 1
    M = len(h_l[0])
    K = len(h_l[0][0])

    # initialization
    E_l = []
    EmixM_l = []
    EmixT_l = []

    for t in range(T+1):
        E_l.append(np.zeros((M, K)))
        EmixM_l.append(np.zeros((M, M, K, K)))
        EmixT_l.append(np.zeros((M, K, K)))

    # message passing initialization
    muf_l = [[np.zeros((K, 1)) for m in range(M)] for t in range(T+1)]
    mub_l = [[np.zeros((K, 1)) for m in range(M)] for t in range(T+1)]

    for m in range(M):
        for k in range(K):
            muf_l[1][m][k] = (logh_l[1][m][k] + (sp_misc.logsumexp(
                logh_l[0][m] + np.reshape(logpi_l[m], (K, 1)) +
                np.reshape(logA_l[m][:, k], (K, 1)))))

    for m in range(M):
        for k in range(K):
            mub_l[T-1][m][k] = (sp_misc.logsumexp(
                logh_l[T][m] + np.reshape(logA_l[m][k, :], (K, 1))))

    # forward-backward message passing
    for t in range(1, T):
        for m in range(M):
            for k in range(K):
                muf_l[t+1][m][k] = logh_l[t+1][m][k] + (sp_misc.logsumexp(
                    np.reshape(logA_l[m][:, k], (K, 1)) +
                    np.copy(muf_l[t][m])))

                mub_l[T-t-1][m][k] = (sp_misc.logsumexp(
                    logh_l[T-t][m] + np.reshape(logA_l[m][k, :], (K, 1)) +
                    np.copy(mub_l[T-t][m])))

    # fill E_l
    for t in range(1, T+1):
        for m in range(M):
            Eint_v = muf_l[t][m] + mub_l[t][m]
            Eint_v += -sp_misc.logsumexp(Eint_v)
            E_l[t][m, :] = np.reshape(np.exp(Eint_v), (K,))

    for m in range(M):
        Eint_v = (logh_l[0][m] + np.reshape(logpi_l[m], (K, 1)) + muf_l[0][m] +
                  mub_l[0][m])
        Eint_v += -sp_misc.logsumexp(Eint_v)
        E_l[0][m, :] = np.reshape(np.exp(Eint_v), (K,))

    # fill EmixM_l
    for t in range(T+1):
        for m1 in range(M):
            for m2 in range(M):
                if m2 == m1:
                    for k in range(K):
                        EmixM_l[t][m1, m1, k, k] = E_l[t][m1, k]
                else:
                    EmixM_l[t][m1, m2, :, :] = (np.dot(
                        np.reshape(E_l[t][m1, :], (K, 1)),
                        np.transpose(
                            np.reshape(E_l[t][m2, :], (K, 1))),))

    # fill EmixT_l
    for t in range(2, T+1):
        for m in range(M):
            Eint_m = np.zeros((K, K))
            for k1 in range(K):
                for k2 in range(K):
                    Eint_m[k1, k2] = (
                        logh_l[t][m][k2] + logA_l[m][k1, k2] +
                        muf_l[t-1][m][k1] + mub_l[t][m][k2])
            Eint_m += -sp_misc.logsumexp(Eint_m)
            EmixT_l[t][m, :, :] = np.exp(Eint_m)

    for m in range(M):
        Eint_m = np.zeros((K, K))
        for k1 in range(K):
            for k2 in range(K):
                Eint_m[k1, k2] = (
                    logh_l[1][m][k2] + logpi_l[m][k1] + logh_l[0][m][k1] +
                    logA_l[m][k1, k2] + muf_l[0][m][k1] + mub_l[1][m][k2])
        Eint_m += -sp_misc.logsumexp(Eint_m)
        EmixT_l[1][m, :, :] = np.exp(Eint_m)

    return [E_l, EmixM_l, EmixT_l]
