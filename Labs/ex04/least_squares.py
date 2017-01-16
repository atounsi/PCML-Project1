# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
 
def least_squares(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    y = y.reshape(-1, 1)
    
    txt = np.transpose(tx)
    XTX = txt.dot(tx)
    
    w = np.linalg.inv(XTX).dot(txt).dot(y)
    
    e = y - tx.dot(w)
    L = 1/(2*y.shape[0]) * np.square(e).sum()
    
    return w,L
