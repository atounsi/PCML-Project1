import numpy as np


def removeNaNByColumn(y, colx, colx_standardized):
    """ given a colX remove all sample that are NaN (-999) """
        
    yx = np.concatenate((y, colx), axis=1)
    yx_standardized = np.concatenate((y, colx_standardized), axis=1)
    
    toremoveIndexes = [i for i in range(len(colx)) if colx[i] == -999]
    
    yx = np.delete(yx, toremoveIndexes, 0)
    yx_standardized = np.delete(yx, toremoveIndexes, 0)
        
    return yx, yx_standardized


def cleanNaNByColumn(y, tx, tx_standardized):
    """ reduce each feature separately removing NaN value (-999) """
    
    tx = np.copy(tx)

    my_y = y.reshape(-1, 1)
    
    clean_data = []
    
    for i in range(tx.shape[1]):

        column = tx[:, i]
        column_standardized = tx_standardized[:, i]
        
        
        yx, yx_standardized = removeNaNByColumn(my_y, column.reshape(-1, 1), column_standardized.reshape(-1, 1))
        
        clean_data.append(yx_standardized)
        
    return clean_data  


def cleanNaNByRow(y, tx, tx_standardized):
    """ Reduce data set removing all sample containing NaN (-999) """
    
    tx = np.copy(tx)
    y = np.copy(y)
    tx_standardized = np.copy(tx_standardized)
  
    toremoveIndexes = np.where(tx == -999)[0]
    y = np.delete(y, toremoveIndexes, 0)
    tx = np.delete(tx, toremoveIndexes, 0)
    tx_standardized = np.delete(tx_standardized, toremoveIndexes, 0)
    
    return y, tx, tx_standardized


def replaceNaNbyColMean(tx):
    """ replace NaN by the column mean """    
    tx = np.copy(tx)
    
    mean = np.nanmean(tx, axis=0)

    for i in range(tx.shape[1]):
        tx[np.isnan(tx[:, i]), i] = mean[i]

    return tx

def replaceNaNbyZeros(tx):
    """ replace NaN by 0 """
    tx = np.copy(tx)
    
    tx[tx == np.nan] = 0.0
    
    return tx

def replaceNaNbyMostFreq(tx):
    """ replace NaN by the most frequent value """
    tx = np.copy(tx)
    x = np.around(tx)
    
    minimum = np.nanmin(x, axis=0)
    maximum = np.nanmax(x, axis=0)
    
    m = np.zeros([1, x.shape[1]])
    
    for i in range(x.shape[1]):
        mostFrequent = np.histogram(x[~np.isnan(x[:, i]), i], bins=np.arange(minimum[i], maximum[i]+2))
        # print("col {} : {}".format(i, mostFrequent))
        m[0, i] = mostFrequent[1][np.argmax(mostFrequent[0])]
    
    
    for i in range(tx.shape[1]):
        tx[np.isnan(tx[:, i]), i] = m[0,i]
        

    return tx

