# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt

from given_proj1_helpers import *
from utilities import *
from clean_data import *
from features_processing import *
from given_costs import *
from implementations import *

def compute_best_parameters(y, tx):
    tx = replaceNaNbyMostFreq(tx)
    y, tx = select_random(y, tx, 50000)
    x_train, y_train, x_test, y_test = split_data(y, tx, 0.7)

    x_train,_,_ = standardizeSameDimension(x_train)
    x_test,_,_ = standardizeSameDimension(x_test)

    d, w, l, s = find_best_degree_and_lambda(y_train, y_test, x_train, x_test, 8)
    
    print('The best score is for degree = ', str(d), ' and lambda = ', str(l))
    print('The resulting score is', s)
    
    return d, l

def compute_prediction(y, tx, tx_test):
    tx = replaceNaNbyMostFreq(tx)
    tx_test = replaceNaNbyMostFreq(tx_test)
    tx,_,_ = standardizeSameDimension(tx)
    tx_test,_,_ = standardizeSameDimension(tx_test)

    print('Adding the LOGARITHM transform of all the features...')
    
    tx = add_new_log_features(tx, np.arange(tx.shape[1]))
    tx_test = add_new_log_features(tx_test, np.arange(tx_test.shape[1]))

    print('Done!')

    print('Adding the SINUS transform of all the features...')
    
    tx = add_sin_features(tx, np.arange(tx.shape[1]))
    tx_test = add_sin_features(tx_test, np.arange(tx_test.shape[1]))

    print('Done!')
    print('Adding the ROOT transform of all the features...')

    tx = add_root_features(tx, np.arange(tx.shape[1]))
    tx_test = add_root_features(tx_test, np.arange(tx_test.shape[1]))
    print('Done!')
    
    tx,_,_ = standardizeSameDimension(tx)
    tx_test,_,_ = standardizeSameDimension(tx_test)
    
    print('Computing the best value for the degree of the polynomial basis...')
    #degree, lambda_ = compute_best_parameters(y, tx)
    degree = 3
    print('Done! Degree =', degree)

    print('Computing the best value for lambda for the ridge regression...')
    lambda_ = 107.226722201
    print('Done! Lambda =', lambda_)
    
    print('Building the new input matrix using the polynomial basis...')
    tx = build_d_poly(tx, degree)
    tx_test = build_d_poly(tx_test, degree)

    print('Done!\n')

    print('---Applying the ridge regression---')
    print('Computing...')
    
    w, loss = ridge_regression(y, tx, lambda_)

    print('Done!')
    
    print('Loss found is', loss)

    print('---Predicting the labels for \'y\'---')
    
    y_pred = predict_labels(w, tx_test)
    
    return y_pred, w, loss
  
def run():
    print('---Program started---\n')
    print('---Importing required datasets---')
    
    ### Import  train data
    DATA_TRAIN_PATH = 'data/train.csv' # TODO: download train data and supply path here 
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH, sub_sample=False)

    print('Training dataset loaded!')

    ## Import Test data
    DATA_TEST_PATH = 'data/test.csv' # TODO: download train data and supply path here 
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

    print('Test dataset loaded!')

    # Replace -999 values by NaN in train and test files.
    tX_NaN = np.copy(tX)
    tX_NaN[tX == -999] = np.nan
    tX_test_NaN = np.copy(tX_test)
    tX_test_NaN[tX_test == -999] = np.nan

    print('Replaced \"-999\" of each attributes by most frequent value of corresponding attribute!\n')

    print('---Prediction computation started---')
    
    y_pred, w, loss = compute_prediction(y, tX_NaN, tX_test_NaN)
    
    OUTPUT_PATH = 'outputs/87_TeamName_output_final.csv' # TODO: fill in desired name of output file for submission

    print('---Creating output file at location:', OUTPUT_PATH)

    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

    print('Done!')
    
run()    