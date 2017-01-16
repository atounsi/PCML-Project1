# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return np.dot(e,e) / (2 * len(e))
    #return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

### Compute the square root of the mean square error
def calculate_rmse(e):
    return np.sqrt(2*calculate_mse(e))


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)
    # return calculate_mae(e)
    # return calculate_rmse(e)
    
def compute_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    y_pred = np.dot(tx,w)
    loss = np.sum(np.logaddexp(0,y_pred) - np.dot(np.transpose(y),y_pred))
    return loss