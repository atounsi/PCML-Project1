# -*- coding: utf-8 -*-
"""some utilities functions."""
import matplotlib.pyplot as plt
import numpy as np

from given_proj1_helpers import *


column_names = ["DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h", "DER_deltaeta_jet_jet","DER_mass_jet_jet","DER_prodeta_jet_jet","DER_deltar_tau_lep","DER_pt_tot","DER_sum_pt","DER_pt_ratio_lep_tau","DER_met_phi_centrality","DER_lep_eta_centrality","PRI_tau_pt","PRI_tau_eta","PRI_tau_phi","PRI_lep_pt","PRI_lep_eta","PRI_lep_phi","PRI_met","PRI_met_phi","PRI_met_sumet","PRI_jet_num","PRI_jet_leading_pt","PRI_jet_leading_eta","PRI_jet_leading_phi","PRI_jet_subleading_pt","PRI_jet_subleading_eta","PRI_jet_subleading_phi","PRI_jet_all_pt"]

## Plot an histogram
def plot_histogram(name, yx):

    #background value are encoded -1 in y
    b =  yx[:,1][yx[:,0]<0]
    #signal value are encoded 1 in y
    s =  yx[:,1][yx[:,0]>0]   
    
    #bins = [x + 0.5 for x in range(0, 6)]
    #, bins = bins
    plt.hist([b, s], color = ['red', 'blue'], label = ['b', 's'])

    plt.ylabel('frequency')
    plt.xlabel('value')
    plt.title(name)
    outputName = "data/Analyse/"+name+".png"
    plt.savefig(outputName)
    plt.legend()
    plt.show()
    
    
def plot_all_histograms(y, tx, clean=True):
    
    y = y.reshape(-1, 1)
    
    for i in range(tx.shape[1]):

        colx = tx[:, i]
        colx = colx.reshape(-1, 1)
            
        yx = np.concatenate((y, colx), axis=1)
    
        if clean : 
            toremoveIndexes = [i for i in range(len(colx)) if colx[i] == -999]
            yx = np.delete(yx, toremoveIndexes, 0)
        
        print(yx.shape)
        plot_histogram("col_{}_{}".format(i, column_names[i]), yx)

def cross_validation_visualization_lambdas(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.figure()
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("mse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation_lambdas.png")

def cross_validation_visualization_degrees(degs, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.figure()
    plt.plot(degs, mse_tr, marker=".", color='b', label='train error')
    plt.plot(degs, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("degree")
    plt.ylabel("mse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation.png")

def bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te):
    """visualize the bias variance decomposition."""
    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)
    plt.plot(
        degrees,
        rmse_tr.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        label='train',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_te.T,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        label='test',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_tr_mean.T,
        'b',
        linestyle="-",
        label='train',
        linewidth=3)
    plt.plot(
        degrees,
        rmse_te_mean.T,
        'r',
        linestyle="-",
        label='test',
        linewidth=3)
    plt.ylim(0.2, 0.7)
    plt.xlabel("degree")
    plt.ylabel("error")
    plt.title("Bias-Variance Decomposition")
    plt.savefig("bias_variance")

        
        

def compute_performance(y, tx, w):
    prediction = predict_labels(w, tx)
    count = np.sum(y == prediction)
    score = count/len(y)
    print('Score is', score)


    
def select_random(y, tX, num_samples, seed=1):
    np.random.seed(seed)
    
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    y2 = y[indices]
    tX2 = tX[indices,:]
    if (num_samples < y2.shape[0]):
        y2 = y2[:num_samples]
        tX2 = tX2[:num_samples,:]
        
    return y2, tX2

def split_data(y, x, ratio, seed=1):
    """split the dataset based on the split ratio."""

    np.random.seed(seed)
    shuffle_indices = np.random.permutation(len(x))
    shuffled_y = y[shuffle_indices]
    shuffled_x = x[shuffle_indices]
    
    index_max = len(x)-1
    i = np.floor(index_max * ratio)
    
    return shuffled_x[0:i], shuffled_y[0:i], shuffled_x[i:index_max], shuffled_y[i:index_max] 
    
    