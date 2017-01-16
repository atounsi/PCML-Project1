###Cross-Validation
import numpy as np
from features_processing import *
from clean_data import *

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    
    return np.array(k_indices)



def cross_validation_poly(y, tx, k_fold, degrees,lambdas,  debug):
    seed = 10
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    loss_tr = []
    loss_te = []
    scores = []
    
    for lambda_ in lambdas:
        for d in degrees:
            print('\nlambda ->', lambda_, ' Degree ->', d, '\n')
            mse_tr_k = 0
            mse_te_k = 0
            score = 0
            for k in range(0, k_fold):
                mse_tr, mse_te, s = compute_fold_k(y, tx, k_indices, k, d, lambda_, debug)
                mse_tr_k += mse_tr
                mse_te_k += mse_te
                score += s
            loss_tr.append(mse_tr_k/k_fold)
            loss_te.append(mse_te_k/k_fold)
            scores.append(score/k_fold)
            print('\nThe results are the following')
            print('Training loss is', mse_tr_k/k_fold)
            print('Test loss is', mse_te_k/k_fold)
            print('Average score is', score/k_fold)
    
    return loss_tr, loss_te, scores

def compute_fold_k(y, x, k_indices, k, degree, lambda_, debug = False):
    
    k_indices_copy = np.copy(k_indices)
        
    x_test = x[k_indices[k]]
    y_test = y[k_indices[k]]

    k_indices_copy = np.delete(k_indices_copy, k, 0)
    k_indices_copy = np.hstack(k_indices_copy)
                
    x_train = x[k_indices_copy]
    y_train = y[k_indices_copy]
        
    #w = least_squares(y_train, x_train_poly)
    #w, l = ridge_regression(y_train, x_train_poly, lambda_)
    #w, l = least_squares_SGD(y_train, x_train_poly, 0.15, 201)
    
    ##############################################################
    
    tx_train = replaceNaNbyMostFreq(x_train)
    tx_test = replaceNaNbyMostFreq(x_test)
    
    
    tx_train = add_inverse_log_features(tx_train, np.arange(30))
    tx_test = add_inverse_log_features(tx_test,  np.arange(30))
    
    #tx_train = add_sin_features(tx_train, np.arange(2))
    #tx_test = add_sin_features(tx_test, np.arange(2))
    
    #tx_train = add_root_features(tx_train, np.arange(10))
    #tx_test = add_root_features(tx_test, np.arange(10))
    
    #tx_train = add_log_features(tx_train, np.arange(2), cst=0.01)
    #tx_test = add_log_features(tx_test, np.arange(2), cst=0.01)
    
    #x_train = add_rational_features(tx_train, np.arange(30))
    #x_test = add_rational_features(tx_test, np.arange(30))
    
    #tx_train = add_Box_Cox_features(tx_train, np.arange(1), lambda_)
    #tx_test = add_Box_Cox_features(tx_test, np.arange(1), lambda_)
    
    tx_train_poly = build_d_poly(tx_train, degree)
    tx_test_poly = build_d_poly(tx_test, degree)
    #tx_train_poly = build_poly_mixed_term(tx_train, degree)
    #tx_test_poly = build_poly_mixed_term(tx_test, degree)
    
    
    w, mse_tr = ridge_regression(y_train, tx_train_poly, lambda_)
    w_test, mse_te = ridge_regression(y_test, tx_test_poly, lambda_)
   
    
    
    y_pred = predict_labels(w, tx_test_poly)
    
    ##############################################################
    
    #mse_tr = compute_loss(y_train, x_train_poly, w)
    #mse_te = compute_loss(y_test, x_test_poly, w)
    
    # Compute the "Kaggle" score => count number of (y_test == prediction(w, x_test))
    #prediction = predict_labels(w, x_test_poly)
    count = np.sum(y_test == y_pred)
    score = count/len(y_test)
    if(debug):
        print('mse_tr at fold', k, 'is', mse_tr)
        print('mse_te at fold', k, 'is', mse_te)
        print('Score at fold', k, 'is', score)
    
    return mse_tr, mse_te, score

