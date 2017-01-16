# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    x = x.reshape(-1)    
    phi = np.zeros((x.shape[0], degree+1))
            
    for n in range(degree+1):
        phi[:, n] = phi[:, n]+np.power(x, n)
       
    return phi
