# -*- coding: utf-8 -*-
"""
Data loading utiliy functions
"""
import scipy.io as sio
import numpy as np
import itertools


from sklearn import datasets
from dkmeans.util import split_chunks

DEFAULT_DATASET = "gaussian"
DEFAULT_THETA = [[0, 1]]
DEFAULT_WINDOW = 50
DEFAULT_M, DEFAULT_N = (1, 2)
DEFAULT_transpose = False
DEFAULT_shuffle = True

SIMULATED_TC_DIR = ("/export/mialab/users/bbaker/projects/djica/tests3"
                    "/IOTA/SIM/22Sep2017/increase_both/"
                    "s2048-n64-nc20-r1/IC.mat")
REAL_TC_DIR = ("/export/mialab/users/bbaker/projects/djica/tests3"
               "/final/s2016-n63-nc50-r1/IC.mat")


def get_dataset(N, dataset=DEFAULT_DATASET, theta=DEFAULT_THETA,
                dfnc_window=DEFAULT_WINDOW, m=DEFAULT_M, n=DEFAULT_N):
    """Convenience function for getting data sets by name
        TODO: Should this be moved to the load data functions? (yes)
    """
    X = None
    if dataset == 'gaussian':
        # TODO!: This line is horrible and hacky, and needs to be fixed
        X = list(itertools.chain.from_iterable([
                            simulated_gaussian_cluster(int(N/len(theta)), t[0],
                                                       t[1], m=m, n=n)
                            for t in theta]))
    elif dataset == 'iris':
        X = datasets.load_iris().data[0:N]
    elif dataset == 'simulated_fmri':
        X = window_all_tc(load_sim_tcs(), dfnc_window, n=N),
    elif dataset == 'real_fmri':
        X = window_all_tc(load_real_tcs(), dfnc_window, n=N)
    m, n = get_data_dims(X)
    X = [np.array(x.reshape([m, n])) for x in X]  # maintain X as a tensor
    return X


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
    return sio.loadmat(REAL_TC_DIR)['Shat'][0]


def window_tc(TC, winsize, transpose=DEFAULT_transpose):
    """ Using a sliding window, find the windows with maximum variance """
    TC_w = []
    TC_v = []
    start = 0
    end = start + winsize
    while end <= TC.shape[0]:
        TT = TC[start:end, :]
        if transpose:
            TT = TT.T
        TC_w += [np.cov(TT.T)]
        TC_v += [np.var(TT.T)]
        start = start+1
        end = start+winsize
    TC_w = TC_w[TC_v.index(np.max(TC_v))]
    return [TC_w]


def window_all_tc(Shat_, winsize, n=0, transpose=DEFAULT_transpose):
    """ Using a sliding window, finding maximally covariant window for
        all subjects
    """
    Shat_w = []
    if n <= 0:
        n = len(Shat_)
    for i in range(n):
        # print(i)
        w = window_tc(Shat_[i], 50, transpose)
        Shat_w += w
    return Shat_w


""" Gaussian Data Functions """


def simulated_gaussian_cluster(N, mu, sigsqr, m, n):
    x = []
    for i in range(N):
        x += [sigsqr * np.random.randn(m, n) + mu]
    return x


""" Data Distribution Functions """


def split_over_nodes(X, s, shuffle=DEFAULT_shuffle):
    """ Split data over s sites, either randomly or sequentially """
    node_distribution = int(np.floor(len(X) / s))
    if shuffle:
        indices = list(np.random.choice(len(X), size=[s, node_distribution]))
    else:
        indices = split_chunks(range(len(X)), node_distribution)
    X_split = [[X[i] for i in chunk] for chunk in indices]
    flat_indices = [index for node in indices for index in node]
    remaining = int(np.ceil(len(X) / s)) - node_distribution
    for index in range(remaining, 0, -1):
        X_split[-1].append(X[-index])
        flat_indices.append(index)
    return X_split, flat_indices
