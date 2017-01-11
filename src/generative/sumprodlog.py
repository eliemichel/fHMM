# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 14:17:37 2016

@author: De Bortoli
"""

import numpy as np


def sumprodlog(vec):
    """
    Compute the logarithm of a sum of the exponential of a vector

    Input arguments:
        vec -- a vector

    Output arguments:
        sumvec, a float: the sum

    Remark:
        Direct implementation without using this function leads to numerical
        errors.
    """
    if vec.size == 0:
        sumvec = -1e308
    else:
        M=np.max(vec)
        sumvec=M+np.log(sum(np.exp(vec-M)))
            
    return sumvec
