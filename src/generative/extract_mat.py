# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 18:24:44 2016

@author: De Bortoli
"""

import numpy as np


def extract_mat(ind, mat, m):
    """
    Extract the vector of size K located at (ind1,...,indm-1,:,indm+1,...,indM)
    in the mat tensor.

    Input arguments:
        ind -- a tuple of size M containing the fixed indices.
        mat -- a tensor of size Kx...xK (M times).
        m -- an integer corresponding to the moving index.

    Output arguments:
        extract_v -- a vector of size K containing the desired vector
    """

    M = mat.ndim
    K = np.shape(mat)[0]
    extract_v = np.zeros(K)

    for k in range(K):
        extract_v[k] = mat[ind[0:m] + tuple([k]) + ind[(m+1):M]]

    return [extract_v]
