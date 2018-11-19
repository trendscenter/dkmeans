# -*- coding: utf-8 -*-
"""

This module was created for testing initial benchmarks of the various
clustering approaches.

"""
import numpy as np
import os

from sklearn import metrics
from time import time


from dkmeans.data import get_dataset
from dkmeans.data import DEFAULT_DATASET, DEFAULT_THETA, DEFAULT_WINDOW
from dkmeans.data import DEFAULT_M, DEFAULT_N
from dkmeans.data import choose_best_centroids
import dkmeans.singleshot as dkss
import dkmeans.multishot as dkms
import dkmeans.kmeans_pooled as kp
from multiprocessing import Process, Queue


METHODS = {
           #'pooled': (kp.main, {}),
            #'singleshot_lloyd': (dkss.main, {'optimization': 'lloyd'}),
            #'singleshot_gradient': (dkss.main, {'optimization': 'gradient'}),
           'multishot_lloyd': (dkms.main, {'optimization': 'lloyd'}),
            #'multishot_gradient': (dkms.main, {'optimization': 'gradient'}),
           }  # STORES the method mains and the kwarg for the corresponding opt
METHOD_NAMES = METHODS.keys()
METRICS = {'silhouette': metrics.silhouette_score,
           }
METRIC_NAMES = METRICS.keys()
DEFAULT_METHOD = "pooled"
DEFAULT_VERBOSE = True


def evaluate_metric(X, labels, metric):
    """
        More helpful for when we have different choices of metrics
    """
    flat_X = [x.flatten() for x in X]
    try:
        return METRICS[metric](flat_X, np.array(labels))
    except ValueError:  # TODO - fix this...
        if len(set(labels)) == 1:
            print(labels)
            print("Bad Clustering - all labels assigned to one cluster")
        return -1


def run_method(X, k, method=DEFAULT_METHOD, subjects=None, **kwargs):
    """
        Run a given method by name
    """
    #    print("Running Method %s" % method)
    start = time()
    if 'init_centroids' in kwargs.keys() and type(
            kwargs['init_centroids']) is dict:
        kwargs['init_centroids'] = kwargs['init_centroids'][method]
    res = METHODS[method][0](X, k, **METHODS[method][1], **kwargs)
    # print(res)
    end = time()
    res['rtime'] = end - start
    res['subjects'] = subjects
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
    subjects = None
    X, subjects = get_dataset(N, dataset=dataset, theta=theta,
                    dfnc_window=dfnc_window, m=m, n=n)
    res = {method: run_method(X, k, subjects=subjects, method=method, **kwargs)
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
    processes = []
    queues = []
    for r in range(R):
        q = Queue()
        p = Process(target=one_run_exp, args=(q, k, N, metrics,
                                              methods, ), kwargs=kwargs)
        p.start()
        processes.append(p)
        queues.append(q)
    for i, pq in enumerate(zip(processes, queues)):
        p, q = pq
        meas, res = q.get()
        p.join()
        for method in methods:
            results[method] += [res[method]]
            for metric in metrics:
                measures[method][metric] += [meas[method][metric]] 
    return measures, results


def one_run_exp(q, k, N, metrics, methods, **kwargs):
    np.random.seed(seed=int(str(os.getpid())))
    meas, res = run_experiment(k, N, metrics=metrics, methods=methods,
                               **kwargs)
    q.put([meas, res])
    # print("Done with experiment k=%d, N=%d in process %s" %
    #      (k, N, os.getpid()))


def dfnc_pipe(k, N, R, s=2):
    print("Running dFNC exemplars k=%d, R=%d, N=%d, s=%d in process %s" %
          (k, R, N, s, os.getpid()))

    meas, res = run_repeated_experiment(R*1, k, N,
                                        dataset='real_fmri_exemplar',
                                        s=s,
                                        verbose=True)
    for method in meas:
        print("saving exemplars")
        np.save('results/exemplar_N%d_s%d_k%d_%s_meas.npy' % (N, s, k, method), meas)
        np.save('results/exemplar_N%d_s%d_k%d_%s_res.npy' % (N, s, k, method), res)
    print("Printing Exemplar Results k=%d, R=%d, N=%d in process %s" %
          (k, R, N, os.getpid()))
    for method in meas:
        for measure in meas[method]:
            print('\t%s, %s: %s' % (method, measure,
                                    np.max(meas[method][measure])))
    for method in meas:
        exemplars = choose_best_centroids('results/exemplar_N%d_s%d_k%d_%s_res.npy'
                                      % (N, s, k, method),
                                      'results/exemplar_N%d_s%d_k%d_%s_meas.npy'
                                      % (N, s, k, method),
                                      list(meas.keys()))
        print("Running dFNC second stage k=%d, R=%d, N=%d in process %s" %
              (k, R, N, os.getpid()))
        measR, resR = run_repeated_experiment(1, k, N,
                                          dataset='real_fmri',
                                          verbose=True,
                                          s=s,
                                          init_centroids=exemplars)
        print("Saving measure results")
        np.save('results/fbirn_N%d_s%d_k%d_%s_meas.npy' % (N, s, k, method), measR)
        print("Saving results")
        np.save('results/fbirn_N%d_s%d_k%d_%s_res.npy' % (N, s, k, method), resR)
    print("Printing Second Stage Results k=%d, R=%d, N=%d in process %s" %
          (k, R, N, os.getpid()))
    for method in measR:
        for measure in measR[method]:
            print('\t%s, %s: %s' % (method, measure,
                                    np.max(measR[method][measure])))


def main():
    """Run a suite of experiments in order"""
    # datar = [  # 'gaussian',
    #            'iris',
    #            'simulated_fmri',
    #           'real_fmri_exemplar'
    #         ]  # datasets to run

    R = 1  # Number of repetitions
    N = 314  # Number of samples

    # Oth experiment is gaussian set with known number of clusters, 3,
    '''
    print("Running known K experiment")
    theta = [[-1, 0.5], [1, 0.5], [2.5, 0.5]]
    meas, res = run_repeated_experiment(R, 3, N, theta=theta, verbose=False)
    np.save('repeat_known_k_meas.npy', meas)
    np.save('repeat_known_k_res.npy', res)
    '''

    # First experiment is increasing k
    # measure the scores and iterations, no runtimes
    '''
    print("Running increasing K experiment")
    k_test = range(4, 5)
    for k in k_test:
        for d in datar:
            print("K: %d; Dataset %s" % (k, d))
            meas, res = run_repeated_experiment(R, k, N,
                                                dataset=d, verbose=False)
            np.save('results/d%s_k%d_meas.npy' % (d, k), meas)
            np.save('results/d%s_k%d_res.npy' % (d, k), res)
    '''
    # Second experiment is increasing N with fixed k
    # Measure the number of iterations and the runtime and the scores
    # TODO: Implement this

    # Third experiment is Increasing number of subjects in simulated
    # Real fMRI data
    # TODO: Implement this

    # Experiment with 300 subjects, 30 sites
    print("Running increasing K experiment dFNC pipeline")
    k_test = [5]
    s = 2
    processes = []
    for k in k_test:
        p = Process(target=dfnc_pipe, args=(k, N, R, s))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == '__main__':
    # TODO Arg-Parsing
    main()
