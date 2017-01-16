# All features (x+log(x)+sin(x)) => degree 6 and lambda 0.01
#The results are the following
#Training loss is 0.276000644122
#Test loss is 0.275124379002
#Average score is 0.813008

# All features (x+sin(x)+log(x)) => degree 6 and lambda 3
#The results are the following
#Training loss is 0.279081834642
#Test loss is 0.279723758129
#Average score is 0.809184

# All features (x+log(x)) => degree 8 and lambda 3
#The results are the following
#Training loss is 0.279556112676
#Test loss is 0.279007235769
#Average score is 0.808912

# All features log(x) => degree 9 and lambda 3
#The results are the following
#Training loss is 0.287528165869
#Test loss is 0.288025898005
#Average score is 0.800308

# Long tail features (x+log(x)) => degree 8 and lambda 3
#The results are the following
#Training loss is 0.290967163128
#Test loss is 0.290852588947
#Average score is 0.79874

# Remove [14, 15, 17, 18, 20] (x+log(x)) => degree 8 and lambda 3
#The results are the following
#Training loss is 0.281962263455
#Test loss is 0.281638369043
#Average score is 0.806316

# Remove [14, 15, 17, 18, 20] log(x) only => degree 8 and lambda 3
#The results are the following
#Training loss is 0.28999040322
#Test loss is 0.289930753044
#Average score is 0.798324

# Remove [14, 15, 17, 18, 20] x+log(x)+sin(x) only => degree 7 and lambda 3
#The results are the following
#Training loss is 0.280987195363
#Test loss is 0.281542662572
#Average score is 0.806604




#y_pred, w, loss = compute_prediction_poly_ridge_regression(y, tX_standardized_NaN, tX_test_standardized_NaN, 4, 3)

#y_pred, w, loss = compute_prediction_expension_poly_ridge_regression(y, tX_standardized_NaN_poly, tX_test_standardized_NaN_poly, degree=4, lambda_=3)

#y_pred, w, loss = compute_prediction_expension_poly_ridge_regression(y, tX_standardized_NaN_poly, tX_test_standardized_NaN_poly, degree=4, lambda_=4)

#y_pred, w, loss = compute_prediction_expension_poly_ridge_regression(y, tX_standardized_NaN_poly, tX_test_standardized_NaN_poly, degree=4, lambda_=5)

#pred = voting_system(y, tX_standardized_NaN, tX_test_standardized_NaN, tX_standardized_NaN_poly, tX_test_standardized_NaN_poly)

#y_pred, w, loss = compute_prediction_poly_4_ridge_regression(y, tX_standardized_NaN_poly, tX_test_standardized_NaN_poly, 4, 3)

#y_pred, w, loss = compute_prediction_log_expension_poly_ridge_regression (y, tX_standardized_NaN_poly, tX_test_standardized_NaN_poly, 4, 3)

#y_pred, w, loss = compute_prediction_log_sin_expension_poly_ridge_regression (y, tX_standardized_NaN_poly, tX_test_standardized_NaN_poly, 6, 3)
#0.80745

#y_pred, w, loss = compute_prediction_log_sin_expension_poly_ridge_regression (y, tX_standardized_NaN_poly, tX_test_standardized_NaN_poly, 6, 0.01)
#Loss is 0.2764834322915573
#Second High score => 0.81448

#y_pred, w, loss = compute_prediction_log_sin_root_expension_poly_ridge_regression (y, tX_standardized_NaN_poly, tX_test_standardized_NaN_poly, 6, 0.01)
#Loss is 0.2761917124390437






def compute_prediction_least_squares(y, tx, tx_test):
    w, loss = least_squares(y, tx)
    
    y_pred = predict_labels(w, tx_test)

    print('compute_prediction_least_squares loss : ', loss)
    
    return y_pred, w, loss

def compute_prediction_least_squares_SGD(y, tx, tx_test, gamma = 0.01, max_iters = 221):
    
    y_pred, initial_w, l = compute_prediction_least_squares(y, tx, tx_test)
    w, loss = least_squares_SGD(y, tx, initial_w, max_iters, gamma)
    
    y_pred = predict_labels(w, tx_test)
    print('compute_prediction_least_squares_SGD gamma={}, max_iters={}: Loss is {}'.format(gamma, max_iters, loss))
    
    return y_pred, w, loss

def compute_prediction_ridge_regression(y, tx, tx_test, lambda_ = 3):
    
    w, loss = ridge_regression(y, tx, lambda_)
    
    y_pred = predict_labels(w, tx_test)
    print('compute_prediction_ridge_regression lambda={}: Loss is {}'.format(lambda_, loss))
    
    return y_pred, w, loss

def compute_prediction_test_01(y, tx, tx_test):
    w, loss = least_squares(y, tx)
    
    y_pred = predict_labels(w, tx_test)
    
    print('compute_prediction_test_01 : Loss is', loss)
    
    return y_pred, w, loss


def compute_prediction_poly_ridge_regression(y, tx, tx_test, degree=4, lambda_=3):
    #indexes = [14, 15, 17, 18, 20] 
    #tx_train = remove_features(tx, indexes)
    #tx_test = remove_features(tx_test, indexes)
    
    #tx_train = choose_features(tx)
    #tx_test = choose_features(tx_test)
    
    tx_train = replaceNaNbyColMean(tx)
    tx_test = replaceNaNbyColMean(tx_test)
    
    tx_train_poly = build_d_poly(tx_train, degree)
    tx_test_poly = build_d_poly(tx_test, degree)
    
    w, loss = ridge_regression(y, tx_train_poly, lambda_)
    
    print('compute_prediction_poly_ridge_regression degree={}, lambda={} : Loss is {}'.format(degree, lambda_, loss))
    
    y_pred = predict_labels(w, tx_test_poly)
    
    return y_pred, w, loss

def compute_prediction_expension_poly_ridge_regression(y, tx, tx_test, degree=4, lambda_=3):
    
    tx_train = replaceNaNbyColMean(tx)
    tx_test = replaceNaNbyColMean(tx_test)
    
    tx_train_poly = build_d_poly(tx_train, degree)
    tx_test_poly = build_d_poly(tx_test, degree)
    
    tx_train_expended = add_functional_features(tx_train_poly, True, False, False)
    tx_test_expended = add_functional_features(tx_test_poly, True, False, False)
    
    w, loss = ridge_regression(y, tx_train_expended, lambda_)
    
    print('compute_prediction_expension_poly_ridge_regression degree={}, lambda={} : Loss is {}'.format(degree, lambda_, loss))
    
    y_pred = predict_labels(w, tx_test_expended)
    
    return y_pred, w, loss


def compute_prediction(y, tx, tx_test, degree=6, lambda_=0.01, inverse_log=False, log=False, new_log=False, root=False, sin=False, poly=False):
    tx_train = replaceNaNbyMostFreq(tx)
    tx_test = replaceNaNbyMostFreq(tx_test)
    
    toPrint = "Prediction ("
    
    if(inverse_log):
        tx_train = add_inverse_log_features(tx_train, np.arange(30))
        tx_test = add_inverse_log_features(tx_test, np.arange(30))
        toPrint += '-log,'
        
    if(log):
        tx_train = add_log_features(tx_train, np.arange(30))
        tx_test = add_log_features(tx_test, np.arange(30))
        toPrint += 'log,'
        
    if(new_log):
        tx_train = add_new_log_features(tx_train, np.arange(30))
        tx_test = add_new_log_features(tx_test, np.arange(30))
        toPrint += 'new_log,'
        
    if(root):
        tx_train = add_root_features(tx_train, np.arange(30))
        tx_test = add_root_features(tx_test,np.arange(30))
        toPrint += 'root,'
        
    if(sin):
        tx_train = add_sin_features(tx_train, np.arange(30))
        tx_test = add_sin_features(tx_test, np.arange(30))
        toPrint += 'sin,'
        
    if(poly):
        tx_train = build_d_poly(tx_train, degree)
        tx_test = build_d_poly(tx_test, degree)
        toPrint += 'poly'
        
    toPrint += ')'
        
    w, loss = ridge_regression(y, tx_train, lambda_)
        
    print(toPrint, 'with degree={} and lambda={} : Loss is {}'.format(degree, lambda_, loss))
    
    y_pred = predict_labels(w, tx_test)
    
    return y_pred, w, loss



def voting_system(y, tx, tx_test):
    
    ##initialization voting result
    vote_result = np.zeros([tx_test.shape[0]])
    
    y_pred, w, loss = compute_prediction(y, tx, tx_test, 6, 0.01, True, False, True, True, True)
    vote_result += y_pred
    
    y_pred, w, loss = compute_prediction(y, tx, tx_test, 6, 0.01, True, False, False, True, True)
    vote_result += y_pred
    
    y_pred, w, loss = compute_prediction(y, tx, tx_test, 6, 0.01, False, True, True, True, True)
    vote_result += y_pred
    
    y_pred, w, loss = compute_prediction(y, tx, tx_test, 6, 0.01, False, True, False, True, True)
    vote_result += y_pred
    
    y_pred, w, loss = compute_prediction(y, tx, tx_test, 6, 0.01, True, True, True, True, True)
    vote_result += y_pred
    
    ## Format votation result !
    vote_result[np.where(y_pred <= 0)]= -1
    vote_result[np.where(y_pred > 0)] = 1
    
    return vote_result






###?????????????????????????????????????????

#y_pred, w1, loss1 = compute_prediction(y, tX_standardized_NaN_poly, tX_test_standardized_NaN_poly, 6, 0.01, True, False, True, False, True, True)
#High score => 0.81506








