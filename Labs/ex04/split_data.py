# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""

import numpy as np

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    
    shuffle_indices = np.random.permutation(np.arange(len(x)))
    shuffled_y = y[shuffle_indices]
    shuffled_x = x[shuffle_indices]
    
    max_i = len(x)-1
    cut_i = np.floor(max_i * ratio)
    
    return shuffled_x[0:cut_i], shuffled_y[0:cut_i],shuffled_x[cut_i:max_i],shuffled_y[cut_i:max_i] 