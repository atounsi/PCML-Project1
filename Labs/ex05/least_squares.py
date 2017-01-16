# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares solution."""
    ttx = np.transpose(tx)
    A = np.dot(ttx,tx)
    b = np.dot(ttx,y)
    w = np.linalg.solve(A,b)
    return w