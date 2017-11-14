#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:37:25 2017

This module was created for testing initial benchmarks of the various
clustering approaches.

@author: bbaker
"""
from dfnc_load_data import window_all_tc, load_sim_tcs, load_real_tcs
from dkmeans_util import simulated_gaussian_cluster
import dkmeans_ss_lloyd as dksl
import dkmeans_ss_gd as dksg
import dkmeans_ms_lloyd as dkml
import dkmeans_ms_gd as dkmg
import kmeans_pooled as kp
import numpy as np
from sklearn import metrics
from sklearn import datasets
from time import time
import itertools


def get_dataset(N, dset='gaussian', theta=[[0, 1]], win=50, m=1, n=2):
    """Convenience function for getting data sets by name
        TODO: Should this be moved to the load data functions? (yes)
    """
    X = None
    if dset == 'gaussian':
        # TODO!: This line is horrible and hacky, and needs to be fixed
        X = list(itertools.chain.from_iterable([
                            simulated_gaussian_cluster(int(N/len(theta)), t[0],
                                              t[1], m=m, n=n) for t in theta]))
    elif dset == 'iris':
        X = datasets.load_iris().data[0:N]
    elif dset == 'simulated_fmri':
        X = window_all_tc(load_sim_tcs(), win, n=N),
    elif dset == 'real_fmri':
        X = window_all_tc(load_real_tcs(), win, n=N)
    return np.array(X)


def evaluate_metric(X, labels, metric=''):
    """
        More helpful for when we have different choices of metrics
    """
    e = {'silhouette': metrics.silhouette_score
         }
    return e[metric](np.array(X), np.array(labels))


def run_method(X, k, s, method='pooled'):
    """
        Run a given method by name
    """
    M = {'pooled': kp.main,
         'ss_ll': dksl.main,
         'ss_gd': dksg.main,
         'ms_ll': dkml.main,
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
                   methods=['pooled', 'ss_ll', 'ss_gd']):
    """
        Run an experiment with a particular choice of
            1. Data set
            2. Parameters k, n, theta, win, m, n, s
            3. metric
            4. method
    """
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
                                     'ss_ll',
                                     'ss_gd',
                                     'ms_ll',
                                     'ms_gd']):
    """
        Run repeated experiments - this function may be unnecesarry and 
        cluttered?
    """
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
    """Run a suite of experiments in order"""
    datar = ['gaussian', 'iris', 'simulated_fmri', 'real_fmri']  # dsets 2 run

    R = 10  # Number of repetitions
    N = 100  # Number of samples

    # Oth experiment is gaussian set with known number of clusters, 3,

    theta = [[-1, 0.5], [1, 0.5], [2.5, 0.5]]
    meas, res = run_repeated_experiment(R, 3, N, theta=theta)
    print(res)
    np.save('repeat_known_k_meas.npy', meas)
    np.save('repeat_known_k_res.npy', res)
    return
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
    # TODO: Implement this

    # Third experiment is Increasing number of subjects in simulated
    # Real fMRI data
    # TODO: Implement this


if __name__ == '__main__':
    main()
