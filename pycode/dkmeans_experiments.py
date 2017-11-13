#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:37:25 2017

@author: bbaker
"""
from dfnc_load_data import window_all_tc, load_real_exemplar, load_exemplar
from dkmeans_util import simulated_gaussian_cluster
import dkmeans_ss_em as dkse
import dkmeans_ss_gd as dksg
import dkmeans_ms_em as dkme
import dkmeans_ms_gd as dkmg
import kmeans_pooled as kp
import numpy as np
from sklearn import metrics
from sklearn import datasets
from time import time
import matplotlib.pyplot as plt
import pickle
import json
import itertools

def get_dataset(N, dset='gaussian', theta=[[0, 1]], win=50, m=1, n=2):
    X = {'gaussian':
         list(itertools.chain.from_iterable([simulated_gaussian_cluster(int(N/len(theta)), t[0],
                                     t[1], m=m, n=n) for t in theta])),
         'iris': datasets.load_iris().data[0:N],
         'simulated_fmri': window_all_tc(load_exemplar(), win, n=N),
         #'real_fmri': window_all_tc(load_real_exemplar(), win, n=N)
         }
    #print(X[dset])
    return np.array([x.flatten() for x in X[dset]])


def evaluate_metric(X, labels, metric=''):
    e = {'silhouette': metrics.silhouette_score
         }
    #print(e[metric](np.array(X), np.array(labels)))
    return e[metric](np.array(X), np.array(labels))


def run_method(X, k, s, method='pooled'):
    M = {'pooled': kp.main,
         'ss_em': dkse.main,
         'ss_gd': dksg.main,
         'ms_em': dkme.main,
         'ms_gd': dkmg.main
         }
    print("\t\tMethod %s" % method)
    start = time()
    res = M[method](X, k, s)
    # print(res)
    end = time()
    res['rtime'] = end - start
    return res


def run_experiment(k, N, dset='gaussian', theta=[[0, 1]],
                   win=50, m=1, n=2, s=2,
                   metrics=['silhouette'],
                   methods=['pooled', 'ss_em', 'ss_gd']):
    X = get_dataset(N, dset=dset, theta=theta, win=win, m=m, n=n)
    res = {method: run_method(X, k, s, method=method)
           for method in methods}
    measures = {res[r]['name']: {metric: evaluate_metric(res[r]['X'],
                                                         res[r]['C'], metric)
                                 for metric in metrics}
                for r in res}
    return measures, res


def run_repeated_experiment(R, k, N, dset='gaussian', theta=[[0, 1]], win=50,
                            m=1, n=2, s=2, metrics=['silhouette'],
                            methods=[
                                     'pooled',
                                     'ss_em',
                                     'ss_gd',
                                     'ms_em',
                                     'ms_gd']):
    measures = {method: {metric: [] for metric in metrics}
                for method in methods}
    results = {method: [] for method in methods}
    for r in range(R):
        print("\tRun %d" % r)
        meas, res = run_experiment(k, N, dset=dset, theta=theta, win=win, m=m,
                                   n=n, s=s, metrics=metrics, methods=methods)
        for method in methods:
            results[method] += [res[method]]
            for metric in metrics:
                measures[method][metric] += [meas[method][metric]]
    return measures, results


def main():
    datar = {'gaussian': [], 'iris': [], 'simulated_fmri': []}

    R = 10
    N = 100

    # Oth experiment is gaussian set with known number of clusters, 3,

    theta = [[-1, 0.5], [1, 0.5], [2.5, 0.5]]
    meas, res = run_repeated_experiment(R, 3, N, theta=theta)
    print(res)
    np.save('repeat_known_k_meas.npy', meas)
    np.save('repeat_known_k_res.npy', res)
    # First experiment is increasing k
    # measure the scores and iterations, no runtimes

    k_test = range(6, 21)
    for k in k_test:
        for d in datar:
            print(k, d)
            meas, res = run_repeated_experiment(R, k, N, dset=d)
            np.save('d%s_k%d_meas.npy' % (d, k), meas)
            np.save('d%s_k%d_res.pkl' % (d, k), res)
    
    # Second experiment is increasing N with fixed k
    # Measure the number of iterations and the runtime and the scores


if __name__ == '__main__':
    main()
