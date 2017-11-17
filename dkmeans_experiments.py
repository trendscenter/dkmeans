# -*- coding: utf-8 -*-
"""

This module was created for testing initial benchmarks of the various
clustering approaches.

"""
import numpy as np

from sklearn import metrics
from time import time


from dkmeans_data import get_dataset
from dkmeans_data import DEFAULT_DATASET, DEFAULT_THETA, DEFAULT_WINDOW
from dkmeans_data import DEFAULT_M, DEFAULT_N
import dkmeans_ss_lloyd as dksl
import dkmeans_ss_gd as dksg
import dkmeans_ms_lloyd as dkml
import dkmeans_ms_gd as dkmg
import kmeans_pooled as kp


METHODS = {'pooled': kp.main,
           'ss_ll': dksl.main,
           'ss_gd': dksg.main,
           'ms_ll': dkml.main,
           'ms_gd': dkmg.main,
          }
METHOD_NAMES = METHODS.keys()
METRICS = {'silhouette': metrics.silhouette_score,
           }
METRIC_NAMES = METRICS.keys()
DEFAULT_METHOD = "pooled"
DEFAULT_NUM_SITES = 2
DEFAULT_K = 2


def evaluate_metric(X, labels, metric):
    """
        More helpful for when we have different choices of metrics
    """
    return METRICS[metric](np.array(X), np.array(labels))


def run_method(X, k, num_sites, method=DEFAULT_METHOD):
    """
        Run a given method by name
    """

    print("\t\tMethod %num_sites" % method)
    start = time()
    res = METHODS[method](X, k, num_sites)
    # print(res)
    end = time()
    res['rtime'] = end - start
    return res


def run_experiment(k, N, dataset=DEFAULT_DATASET, theta=DEFAULT_THETA,
                   dfnc_window=DEFAULT_WINDOW, m=DEFAULT_M, n=DEFAULT_N,
                   num_sites=DEFAULT_NUM_SITES,
                   metrics=['silhouette'],
                   methods=METHODS):
    """
        Run an experiment with a particular choice of
            1. Data set
            2. Parameters k, n, theta, dfnc_window, m, n, num_sites
            3. metric
            4. method
    """
    X = get_dataset(N, dataset=dataset, theta=theta, dfnc_window=dfnc_window, m=m, n=n)
    res = {method: run_method(X, k, num_sites, method=method)
           for method in methods}
    measures = {res[r]['name']: {metric: evaluate_metric(res[r]['X'],
                                                         res[r]['C'], metric)
                                 for metric in metrics}
                for r in res}
    return measures, res


def run_repeated_experiment(R, k, N, dataset=DEFAULT_DATASET,
                            theta=DEFAULT_THETA, dfnc_window=DEFAULT_WINDOW,
                            m=DEFAULT_M, n=DEFAULT_N,
                            num_sites=DEFAULT_NUM_SITES, metrics=METRIC_NAMES,
                            methods=METHOD_NAMES):
    """
        Run repeated experiments - this function may be unnecesarry and 
        cluttered?
    """
    measures = {method: {metric: [] for metric in metrics}
                for method in methods}
    results = {method: [] for method in methods}
    for r in range(R):
        print("\tRun %d" % r)
        meas, res = run_experiment(k, N, dataset=dataset, theta=theta, dfnc_window=dfnc_window, m=m,
                                   n=n, num_sites=num_sites, metrics=metrics, methods=methods)
        for method in methods:
            results[method] += [res[method]]
            for metric in metrics:
                measures[method][metric] += [meas[method][metric]]
    return measures, results


def main():
    """Run a suite of experiments in order"""
    datar = ['gaussian', 'iris', 'simulated_fmri', 'real_fmri']  # datasets 2 run

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
            meas, res = run_repeated_experiment(R, k, N, dataset=d)
            np.save('d%s_k%d_meas.npy' % (d, k), meas)
            np.save('d%s_k%d_res.pkl' % (d, k), res)

    # Second experiment is increasing N with fixed k
    # Measure the number of iterations and the runtime and the scores
    # TODO: Implement this

    # Third experiment is Increasing number of subjects in simulated
    # Real fMRI data
    # TODO: Implement this


if __name__ == '__main__':
    # TODO Arg-Parsing
    main()
