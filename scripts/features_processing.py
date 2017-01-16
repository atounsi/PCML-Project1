### Imports
import numpy as np
from implementations import *
from utilities import *
from given_proj1_helpers import *


#########################################################
### Standarization
#########################################################

def standardize(x, mean_x=None, std_x=None):
    """ 
    Standardize the original data set
    this is the given method, adding a column of 1s at the begining 
    """

    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    
    x = np.hstack((np.ones((x.shape[0],1)), x))
    return x, mean_x, std_x


def standardizeSameDimension(x, mean_x=None, std_x=None):
    """
    Standardize the original data set.
    To be used when we apply phi (polynomial). 
    This method, does not append a '1' column
    """

    x = np.copy(x)
    
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
        
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    
    return x, mean_x, std_x

def standardize_NaN(x, mean_x=None, std_x=None):
    """
    Standardize the original data set (where -999 has been replaced by NaN).
    This methods appends a column of 1s in front of x
    """

    if mean_x is None:
        mean_x = np.nanmean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.nanvar(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]

    x = np.hstack((np.ones((x.shape[0],1)), x))
    return x, mean_x, std_x


def standardize_NaN_poly(x, mean_x=None, std_x=None):
    """
    Standardize the original data set (where -999 has been replaced 
    by NaN).
    To be used when we apply phi (polynomial). 
    This method, does not append a '1' column
    """

    if mean_x is None:
        mean_x = np.nanmean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.nanvar(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]

    return x, mean_x, std_x

#########################################################
### Feature Expension
#########################################################
########## Polynomial expension
#########################################################
def build_poly(x, degree):
    """ builds polynomial feature matrix for 1 feature """
    poly = np.zeros([len(x), degree])
    for j in range(degree):
        poly[:,j] = pow(x,j+1)
    return poly


def build_d_poly(x, degree):
    """ builds polynomial feature matrix for d features """
    n = x.shape[0]
    d = x.shape[1]
    x_new = np.ones([n,1])
    for i in range(d):
        x_new = np.c_[x_new,build_poly(x[:,i],degree)]
    return x_new


def mix_terms(a, b):
    """ 
    Mix term a with terme b and return new columns
    if a = (x, y, z) and b = (u, v, w) we got
    (xu, xv, xw, yu, yv, yw, zu, zv, zw)
    a mixed with itself give : 
    xx, xy, xz, yx, yy, yz, zx, zy, zz

    Positive point : we add mixed term for polynomial expension
    Negative point : we get redoundancy
    """

    featuresA = a.shape[1]
    featuresB = b.shape[1]
    
    colA0 = a[:, 0]
    colB0 = b[:, 0]
    mixed = colA0*colB0
    
    for i in range(featuresA):
        colAi = a[:, i]
        
        for j in range(featuresB):
            if not (i == 0 and j==0) :
                colBj = b[:, j]
                mixed = np.c_[mixed, colAi*colBj]
    
    return mixed


def build_poly_mixed_term(x, colToMix, degree=2):
    """
    The idea is to get a polynomial of x of degree "degree" 
    with mixed features
    x = (a, b,c) with degree 3 give : 
    (a, b,c)*(a, b,c)*(a, b,c) 
    = (a, b,c)*(aa, ab, ac, ba, bb, bc, ca, cb, cc)

    The main problem the number of feature increase exponnentially 
    with degrees. And as seen in mix_term, we have redundancy
    """

    if len(colToMix) == 0 :
        raise AttributeError

    phi = x[:, colToMix]
    
    for d in range(degree-1):        
        phi = mix_terms(x[:, colToMix], phi)
        
        return np.c_[x, phi]


#########################################################
########## Functional expension
#########################################################

def get_sin_features(x, colIndices):
    """
    Return the sin(x) features for selected features of x
    with colIndices. 
    This method doesnt concatanate sinus features with x
    see : add_sin_features 
    """

    if len(colIndices) == 0 : 
        raise AttributeError
    
    col = x[:, colIndices[0]]
    x_new = np.sin(col)
    
    for i in range(1, len(colIndices)):
        col = x[:, colIndices[i]]
        x_new = np.c_[x_new, np.sin(col)]
    
    return x_new



def get_log_features(x, colIndices, cst=0.03):
    """
    Return the log(|x|+cst) features for selected features of x
    with colIndices.  
    This method doesnt concatanate log features with x
    see : add_log_features 
    """
    
    if len(colIndices) == 0 : 
        raise AttributeError
    
    col = x[:, colIndices[0]]
    col = np.absolute(col)
    
    x_new = np.log(col+cst)
    
    for i in range(1, len(colIndices)):
        col = x[:, colIndices[i]]
        col = np.absolute(col)
        x_new = np.c_[x_new, np.log(col+cst)]
    
    return x_new


def get_inverse_log_features(x, colIndices):
    """
    Return the -log(x) features for selected features of x
    with colIndices. 
    This method doesnt concatanate log features with x
    this methode leave invalid log values, and not apply log(x)
    on them. 
    see : add_inverse_log_features 
    """

    if len(colIndices) == 0 : 
        raise AttributeError
    
    col = x[:, colIndices[0]]
    x_new = np.where(col > 0, -np.log(col), col)
    
    for i in range(1, len(colIndices)):
        col = x[:, colIndices[i]]
        x_new = np.c_[x_new, np.where(col > 0, -np.log(col), col)]
    
    return x_new


def get_new_log_features(x, colIndices):
    """
    Return the sign(x) * log(|x|+1) features for selected features of x
    with colIndices. 
    This method doesnt concatanate log features with x
    see : add_new_log_features 
    """

    if len(colIndices) == 0 : 
        raise AttributeError
    
    col = x[:, colIndices[0]]
    x_new = np.sign(col)*np.log(np.absolute(col)+1)
    
    for i in range(1, len(colIndices)):
        col = x[:, colIndices[i]]
        x_new = np.c_[x_new, np.sign(col)*np.log(np.absolute(col)+1)]
    
    return x_new


def get_root_features(x, colIndices):
    """
    Return the sqrt(x) features for selected features of x
    with colIndices. 
    this methode leave invalid root values, and not apply sqrt(x)
    This method doesnt concatanate root features with x
    see : add_root_features 
    """    

    if len(colIndices) == 0 : 
        raise AttributeError
    
    col = x[:, colIndices[0]]
    
    x_new = np.where(col >= 0, np.sqrt(col), col)
    
    
    for i in range(1, len(colIndices)):
        col = x[:, colIndices[i]]
        x_new = np.c_[x_new, np.where(col >= 0, np.sqrt(col), col)]
    
    return x_new



def add_log_features(x, colIndices, cst=0.03):
    """ Concatanate log feature to given x"""
    return np.c_[x, get_log_features(x, colIndices, cst)]
        

def add_inverse_log_features(x, colIndices):
    """ Concatanate inverse log feature to given x"""
    return np.c_[x, get_inverse_log_features(x, colIndices)]

def add_new_log_features(x, colIndices):
    """ Concatanate new_log feature to given x"""
    return np.c_[x, get_new_log_features(x, colIndices)]

def add_sin_features(x, colIndices):
    """ Concatanate sin feature to given x"""
    return np.c_[x, get_sin_features(x, colIndices)]

def add_root_features(x, colIndices):
    """ Concatanate root feature to given x"""
    return np.c_[x, get_root_features(x, colIndices)]
    

#########################################################
### Feature Reducing
#########################################################
def remove_features(x, indexes):
    """Remove the indexes column of x and return the reduced x"""

    return np.delete(x, indexes, 1)

def keep_features(x, indexes):
    """Keep only the indexes column of x and return the reduced x"""   
    toRemove = np.arange(x.shape[1]).reshape(1, -1) 
    toRemove = np.delete(toRemove, indexes, 1)
    
    return np.delete(x, toRemove, 1)


def pca(tX, ratio):
    """ """
    tx = np.copy(tX)
    
    numberOfSamples = tx.shape[0]
    numberOfFeatures = tx.shape[1]
    
    # Substract the mean for each feature.
    for feature in range(numberOfFeatures):
        feature_mean = np.mean(tx[:,feature])
        for sample in range(numberOfSamples):
             tx[sample, feature] = tx[sample, feature] - feature_mean
    
    # Compute the covariance matrix.
    C = np.dot(np.transpose(tx), tx)
    
    # Compute the eigenvalues and eigenvectors of the covariance matrix.
    eigenvalues, eigenvectors = np.linalg.eig(C)
    
    # Order the eigenvectors by sorting the corresponding eigenvalues in decreasing order.
    ind = np.argsort(eigenvalues)
    ind = ind[::-1]
    eigenvectors = eigenvectors[:,ind]
    
    k = 0
    ref_sum = 0
    total_sum = np.sum(eigenvalues)
    while (ref_sum < ratio*total_sum):
        k = k+1
        ref_sum = ref_sum + eigenvalues[k-1]
        
    print("Number of features: ", k)
    
    return eigenvectors[:,:k], k


#########################################################
### Finding best Parameter
#########################################################

def find_best_degree_and_lambda(y_train, y_test, tx_train, tx_test, degree):
    lambdas = np.logspace(-3, 2, 100)
    
    number_of_features = tx_train.shape[1]
    
    best_degree = 0
    best_w = 0
    best_lambda = 0
    best_score = 0
    
    tx_train = add_inverse_log_features(tx_train, np.arange(number_of_features))
    tx_test = add_inverse_log_features(tx_test, np.arange(number_of_features))
    
    tx_train = add_new_log_features(tx_train, np.arange(number_of_features))
    tx_test = add_new_log_features(tx_test, np.arange(number_of_features))
    
    tx_train = add_sin_features(tx_train, np.arange(number_of_features))
    tx_test = add_sin_features(tx_test, np.arange(number_of_features))
    
    tx_train,_,_ = standardizeSameDimension(tx_train)
    tx_test,_,_ = standardizeSameDimension(tx_test)
    
    for d in range(1, degree):
        
        x_train = build_d_poly(tx_train, d)
        x_test = build_d_poly(tx_test, d)
    
        for lambda_ in lambdas:
            
            w, mse_tr = ridge_regression(y_train, x_train, lambda_)
            
            y_pred = predict_labels(w, x_test)

            count = np.sum(y_test == y_pred)
            score = count / len(y_test)

            if score > best_score:
                best_w = w
                best_score = score
                best_lambda = lambda_
                best_degree = d
        
        print('Degree =', str(d), '=> best lambda =', str(best_lambda), 'and score is', str(best_score))
    
    return best_degree, best_w, best_lambda, best_score


