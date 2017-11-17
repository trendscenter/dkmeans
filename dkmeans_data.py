# -*- coding: utf-8 -*-
"""
Data loading utiliy functions
"""
import scipy.io as sio
import numpy as np

DEFAULT_DATASET = "gaussian"
DEFAULT_THETA = [[0, 1]]
DEFAULT_WINDOW = 50
DEFAULT_M, DEFAULT_N = (1, 2)


def get_dataset(N, dataset=DEFAULT_DATASET, theta=DEFAULT_DATASET,
                dfnc_window=DEFAULT_WINDOW, m=DEFAULT_M, n=DEFAULT_N):
    """Convenience function for getting data sets by name
        TODO: Should this be moved to the load data functions? (yes)
    """
    X = None
    if dataset == 'gaussian':
        # TODO!: This line is horrible and hacky, and needs to be fixed
        X = list(itertools.chain.from_iterable([
                            simulated_gaussian_cluster(int(N/len(theta)), t[0],
                                              t[1], m=m, n=n) for t in theta]))
    elif dataset == 'iris':
        X = datasets.load_iris().data[0:N]
    elif dataset == 'simulated_fmri':
        X = window_all_tc(load_sim_tcs(), dfnc_window, n=N),
    elif dataset == 'real_fmri':
        X = window_all_tc(load_real_tcs(), dfnc_window, n=N)
    return np.array(X)

""" DFNC Data Functions"""
def load_sim_tcs():
    """ Load simulated timecourses after djICA preprocessing """
    TCDir = ("/export/mialab/users/bbaker/projects/djica/tests3"
             "/IOTA/SIM/22Sep2017/increase_both/s2048-n64-nc20-r1/IC.mat")
    TC = sio.loadmat(TCDir)
    return TC['Shat_'][0]


def load_real_tcs():
    """ Load real timecourses after djICA preprocessing """
    TCDir = ("/export/mialab/users/bbaker/projects/djica/tests3"
             "/final/s2016-n63-nc50-r1/IC.mat")
    TC = sio.loadmat(TCDir)
    return TC['Shat'][0]


def window_tc(TC, winsize, transpose=False):
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


def window_all_tc(Shat_, winsize, n=0, transpose=False):
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
