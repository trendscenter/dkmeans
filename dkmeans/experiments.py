# -*- coding: utf-8 -*-
"""

This module was created for testing initial benchmarks of the various
clustering approaches.

"""
import numpy as np
import logging

from sklearn import metrics
from time import time

from dkmeans.data import get_dataset
from dkmeans.data import DEFAULT_DATASET, DEFAULT_THETA, DEFAULT_WINDOW
from dkmeans.data import DEFAULT_M, DEFAULT_N
import dkmeans.singleshot as dkss
import dkmeans.multishot as dkms
import dkmeans.kmeans_pooled as kp


METHODS = {'pooled': (kp.main, {}),
           'singleshot_lloyd': (dkss.main, {'optimization': 'lloyd'}),
           'singleshot_gradient': (dkss.main, {'optimization': 'gradient'}),
           'multishot_lloyd': (dkms.main, {'optimization': 'lloyd'}),
           'multishot_gradient': (dkms.main, {'optimization': 'gradient'}),
           }  # STORES the method mains and the kwarg for the corresponding opt
METHOD_NAMES = METHODS.keys()
METRICS = {'silhouette': metrics.silhouette_score,
           }
METRIC_NAMES = METRICS.keys()
DEFAULT_METHOD = "pooled"
DEFAULT_VERBOSE = True

logger = logging.getLogger('dkmeans')
logger.setLevel(logging.INFO)


def evaluate_metric(X, labels, metric):
    """
        More helpful for when we have different choices of metrics
    """
    flat_X = [x.flatten() for x in X]
    try:
        return METRICS[metric](flat_X, np.array(labels))
    except ValueError:  # TODO - fix this...
        if len(set(labels)) == 1:
            logger.info("Bad Clustering - all labels assigned to one cluster")


def run_method(X, k, method=DEFAULT_METHOD, **kwargs):
    """
        Run a given method by name
    """
    logger.info("Running Method %s" % method)
    start = time()
    res = METHODS[method][0](X, k, **METHODS[method][1], **kwargs)
    end = time()
    res['rtime'] = end - start
    return res


def run_experiment(k, N, dataset=DEFAULT_DATASET, theta=DEFAULT_THETA,
                   dfnc_window=DEFAULT_WINDOW, m=DEFAULT_M, n=DEFAULT_N,
                   metrics=METRIC_NAMES,
                   methods=METHOD_NAMES, **kwargs):
    """
        Run an experiment with a particular choice of
            1. Data set
            2. Data Parameters k, n, theta, dfnc_window, m, n
            3. metric
            4. method
            5. Method parameters passed in kwargs
    """
    X = get_dataset(N, dataset=dataset, theta=theta,
                    dfnc_window=dfnc_window, m=m, n=n)
    res = {method: run_method(X, k, method=method, **kwargs)
           for method in methods}
    measures = {res[r]['name']: {metric: evaluate_metric(res[r]['X'],
                                 res[r]['cluster_labels'], metric)
                                 for metric in metrics} for r in res}
    return measures, res


def run_repeated_experiment(R, k, N, metrics=METRIC_NAMES,
                            methods=METHOD_NAMES, **kwargs):
    """
        Run repeated experiments - this function may be unnecesarry and
        cluttered?
    """
    measures = {method: {metric: [] for metric in metrics}
                for method in methods}
    results = {method: [] for method in methods}
    for r in range(R):
        logger.info("Repeated Run Number %d" % r)
        meas, res = run_experiment(k, N, metrics=metrics, methods=methods,
                                   **kwargs)
        for method in methods:
            results[method] += [res[method]]
            for metric in metrics:
                measures[method][metric] += [meas[method][metric]]
    return measures, results


def main():
    """Run a suite of experiments in order"""
    datar = ['gaussian', 'iris',
             # 'simulated_fmri', 'real_fmri'
             ]  # datasets 2 run

    R = 1  # Number of repetitions
    N = 10  # Number of samples

    # Oth experiment is gaussian set with known number of clusters, 3,
    logger.info("Running known K experiment")
    theta = [[-1, 0.5], [1, 0.5], [2.5, 0.5]]
    meas, res = run_repeated_experiment(R, 3, N, theta=theta, verbose=False)
    np.save('repeat_known_k_meas.npy', meas)
    np.save('repeat_known_k_res.npy', res)
    # First experiment is increasing k
    # measure the scores and iterations, no runtimes
    logger.info("Running increasing K experiment")
    k_test = range(2, 4)
    for k in k_test:
        for d in datar:
            logger.info("K: %d; Dataset %s" % (k, d))
            meas, res = run_repeated_experiment(R, k, N,
                                                dataset=d, verbose=False)
            np.save('d%s_k%d_meas.npy' % (d, k), meas)
            np.save('d%s_k%d_res.npy' % (d, k), res)
    # Second experiment is increasing N with fixed k
    # Measure the number of iterations and the runtime and the scores
    # TODO: Implement this

    # Third experiment is Increasing number of subjects in simulated
    # Real fMRI data
    # TODO: Implement this


if __name__ == '__main__':
    # TODO Arg-Parsing
    main()
