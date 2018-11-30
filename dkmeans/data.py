# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
"""
Data loading utiliy functions
"""
import scipy.io as sio
import numpy as np
import itertools


from sklearn import datasets
from dkmeans.util import split_chunks, local_maxima

DEFAULT_DATASET = "real_fmri_exemplar"
DEFAULT_THETA = [[0, 1]]
DEFAULT_WINDOW = 22
DEFAULT_M, DEFAULT_N = (1, 2)
DEFAULT_K = 5

SIMULATED_TC_DIR = ("/export/mialab/users/bbaker/projects/djica/tests3"
                    "/IOTA/SIM/22Sep2017/increase_both/"
                    "s2048-n64-nc20-r1/IC.mat")
REAL_TC_DIR = ("./dkm_in.mat")


def get_dataset(N, dataset=DEFAULT_DATASET, theta=DEFAULT_THETA,
                dfnc_window=DEFAULT_WINDOW, m=DEFAULT_M, n=DEFAULT_N,
                k=DEFAULT_K):
    """Convenience function for getting data sets by name
        TODO: Should this be moved to the load data functions? (yes)
    """
    X = None
    i = None
    if dataset == 'gaussian':
        X, y = datasets.make_blobs(n_samples=N,n_features=n,centers=k)
        X = [x.reshape(1,n) for x in X]
    elif dataset == 'iris':
        X = datasets.load_iris().data[0:N]
    elif dataset == 'simulated_fmri':
        X, i = window_all_tc(load_sim_tcs(), dfnc_window, n=N),
    elif dataset == 'simulated_fmri_exemplar':
        X, i = window_all_tc(load_sim_tcs(), dfnc_window, n=N, exemplar=True)
    elif dataset == 'real_fmri':
        X, i = window_all_tc(load_real_tcs(), dfnc_window, n=N, transpose=True)
    elif dataset == 'real_fmri_exemplar':
        X, i = window_all_tc(load_real_tcs(), dfnc_window, n=N, exemplar=True,
                             transpose=True)
    m, n = get_data_dims(X)
    Xr = []
    for x in X:
        m, n = np.array(x).shape
        Xr.append(np.array(x.reshape([m, n])))
    # X = [np.array(x.reshape([m, n])) for x in X]  # maintain X as a tensor
    X = Xr
    return(X, i)


def get_data_dims(X):
    try:
        [m, n] = X[0].shape
    except ValueError:
        [m, n] = 1, X[0].size
    return m, n


""" DFNC Data Functions"""


def load_sim_tcs():
    """ Load simulated timecourses after djICA preprocessing """
    return sio.loadmat(SIMULATED_TC_DIR)['Shat_'][0]


def load_real_tcs():
    """ Load real timecourses after djICA preprocessing """
    try:
        return sio.loadmat(REAL_TC_DIR)['Shat'][0]
    except KeyError:
        try:
            return sio.loadmat(REAL_TC_DIR)['Shat_'][0]
        except KeyError:
            print("Incorrect key")
            pass


def subject_window_partition(all_win, shat, winsize):
    """given a vector divided by all subject windows, and a list of all subject TCs
       re-partition the vector in a subject-specific list of windows"""
    subj_win = [(s.shape[1] - winsize + 1) for s in shat]
    return np.split(all_win, np.cumsum(subj_win)-subj_win[0])


def window_tc(TC, winsize, transpose=False, exemplar=False):
    """ Using a sliding window, find the windows with maximum variance """
    TC_w = []
    TC_v = []
    start = 0
    end = start + winsize
    if transpose:
        TC = TC.T
    while end <= TC.shape[0]:
        TT = TC[start:end, :]
        #TT = TT.T - np.mean(TT.T)
        COV = np.corrcoef(TT.T)
        TC_w += [COV]
        TC_v.append(np.var(TT))
        start += 1
        end = start+winsize
    if exemplar:
        mm, LM = local_maxima(np.array(TC_v))
        TC_w = [TC_w[i] for i in LM]
    return TC_w


def window_all_tc(Shat_, winsize, n=0, transpose=False, exemplar=False):
    """ Using a sliding window, finding maximally covariant window for
        all subjects
    """
    Shat_w = []
    Subject_labels = []
    if n <= 0:
        n = len(Shat_)
    for i in range(n):  # TODO put this into a comprehension
        w = window_tc(Shat_[i], winsize,
                      transpose=transpose, exemplar=exemplar)
        Shat_w += w
        Subject_labels += [i for wi in w]
    return(Shat_w, Subject_labels)


""" Gaussian Data Functions """


def simulated_gaussian_cluster(N, mu, sigsqr, m, n):
    return [sigsqr * np.random.randn(m, n) + mu for i in range(N)]


""" Data Distribution Functions """


def split_over_nodes(X, s, shuffle=True):
    """ Split data over s sites, either randomly or sequentially
        old - bad - doesn't work
    """
    node_distribution = int(np.floor(len(X) / s))
    if shuffle:
        indices = list(np.random.choice(len(X), size=[s, node_distribution]))
    else:
        indices = list(split_chunks(list(range(len(X))), node_distribution))
    if len(indices) > s:  # TODO: FIX BAD WORKAROUND
        tmp = [si for sub in indices[s:] for si in sub]
        indices = indices[:s]
        indices[s-1] += tmp
    X_split = [[X[i] for i in chunk] for chunk in indices]
    flat_indices = [index for node in indices for index in node]
    remaining = int(np.ceil(len(X) / s)) - node_distribution
    for index in range(remaining, 0, -1):
        X_split[-1].append(X[-index])
        flat_indices.append(index)
    return X_split, flat_indices


def choose_best_centroids(res_file, meas_file, methods, measure='silhouette'):
    if type(methods) is not list:
        methods = list(methods)
    results = {method: None for method in methods}
    for method in methods:
        meas = np.load(meas_file)
        meas = meas.item()
        meas = meas[method][measure]
        res = np.load(res_file)
        res = res.item()
        res = res[method]
        best_index = meas.index(np.max(meas))
        results[method] = res[best_index]['centroids']
    return(results)
