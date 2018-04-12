import numpy as np
from brian2 import *
# utilities

def regularity(spike_trains,indices, time_treshhold = 0):
    # Simply applies ISI_CV to many (all) neurons and returns mean value
    
    cvs = [ISI_CV(st, time_treshhold) for st in spike_trains.values()]
    return np.nanmean(cvs[indices[0]:indices[1]])

def ISI_CV(st, time_treshhold = 0):
    # st is an array of spike times for one neuron
    # time_threshold is moment when we start counting spikes
    # this function computes ICI_CV for this neuron
    
    st_after_threshhold = [s for s in st if s > time_treshhold]
    
    # go from spike times to spike time differences
    ISI = [st_after_threshhold[i] - st_after_threshhold[i -1] for i in range(1, len(st_after_threshhold))]
    
    # get statistics
    std = np.std(ISI)
    mean = np.mean(ISI)
    return std / mean

def correlation(time, st1, st2):
    # Finds correlation for filtered spike times (convolved with biexp)
    
    fst1, fst2= filter_spike_train(st1, time), filter_spike_train(st2, time)
    cov_mat = np.cov(fst1, fst2)
    return cov_mat[1, 0] / np.sqrt(cov_mat[0, 0]*cov_mat[1, 1])

def filter_spike_train(st, time):
    # Convolves biexp. kernel with spike times for one neuron
    
    s_rd = np.round(st/ms, 1)
    S = [i in s_rd and 1 or 0 for i in time]
    kernel = bi_exp(np.arange(-1000, 1000, .1))
    f = np.convolve(S, kernel, 'same')
    return f

def bi_exp(t, tau1 = 50, tau2 = 200):
    # Bi-exponential kernel
    
    return 1/tau1 * exp(-np.abs(t)/tau1) - 1/tau2 * exp(-np.abs(t)/tau2)