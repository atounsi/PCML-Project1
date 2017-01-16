# -*- coding: utf-8 -*-
"""A function to compute the cost."""
import numpy as np

def compute_mse(y, tx, beta):
    """compute the loss by mse."""
        
    y = y.reshape(-1, 1)
    beta = beta.reshape(-1, 1)
        
    e = y - tx.dot(beta)
    mse = np.square(e).sum() / (2 * len(e))
    return mse


def compute_rmse(y, tx, w):
    return np.sqrt(2*compute_mse(y, tx, w))


