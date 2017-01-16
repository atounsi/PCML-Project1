# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    y = y.reshape(-1, 1)

    txt = np.transpose(tx)
    
    XTXL = txt.dot(tx)+ lamb*np.identity(tx.shape[1])    
  
    w = np.linalg.inv(XTXL).dot(txt).dot(y)
    
    return w