#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Unit testing for the dkmeans repository
"""

import numpy as np
import dkmeans.remote_computations as remote


DEFAULT_m = 2
DEFAULT_n = 2
DEFAULT_k = 2
DEFAULT_s = 3
DEFAULT_N = 11


def _k_cluster_labels(N, k):
    '''utility function - arbitrarily fill an N-len array with k labels'''
    C = []
    for ki in range(k):
        C.extend([ki for i in range(int(N/k))])
    while len(C) < N:
        C.append(k - 1)
    return C


def test_remote_aggregate_clusters():
    '''
        Tests the remote aggregation of clusters by merging clusters of closest
        distance.
    '''
    m, n, k, s = DEFAULT_m, DEFAULT_n, DEFAULT_k, DEFAULT_s
    centroids = [[np.ones([m, n]) for ki in range(k)] for si in range(s)]
    expected = [np.ones([m, n]) for ki in range(k)]
    actual = remote.aggregate_clusters(centroids)
    assert all([np.array_equal(a, e) for a, e in zip(expected, actual)])


def test_remote_aggregate_sum():
    '''
        Test the remote aggregate_sum function by summing two numerical arrays
        distributed at different sites.
    '''
    objects = [[np.array([0]), np.array([1])],
               [np.array([1]), np.array([0])]
               ]
    expected = [np.array([1]), np.array([1])]
    actual = remote.aggregate_sum(objects)
    np.testing.assert_array_equal(expected, actual)


def test_remote_closest_centroids():
    '''
        Test the closest_centroid function by generating matrices of ones
        on each site. The generated array should return a tuple where each
        index is associated with the subsequent index.
    '''
    m, n, k, s = DEFAULT_m, DEFAULT_n, DEFAULT_k, DEFAULT_s
    centroids = [[np.ones([m, n])
                 for ki in range(k)] for si in range(s)]
    actual = remote.closest_centroids(centroids)
    expected = [(i, i+1) for i in range(0, k*s-1, 2)]
    if k*s % 2 != 0:
        expected.append((k*s-1, k*s-1))
    assert actual == expected


"""
TODO: finish unit testing
def test_local_compute_mean():
    m, n, k, s, N = DEFAULT_PARAMS
    X = [np.ones([m, n]) for Ni in range(N)]
    C = _k_cluster_labels(N, k)
    expected = [np.ones([m, n]) for ki in range(k)]
    actual = local.compute_mean(X, C, k)
    np.testing.assert_array_equal(expected, actual)

def test_local_mean_step():
    m, n, k, s, N = DEFAULT_PARAMS
    C = _k_cluster_labels(N, k)
    X = [np.ones([m, n]) for Ni in range(N)]
    centroids = [[np.ones([m, n]) for ki in range(k)] for si in range(s)]
    local_means = local.compute_mean(X, C, k)
    expected = ([lmean for lmean in local_means], centroids)
    actual = local.mean_step(local_means, centroids)
    assert all([np.array_equal(a, e) for a, e in zip(expected, actual)]))

def test_local_compute_clustering():
    m, n, k, s, N = DEFAULT_PARAMS
    C = _k_cluster_labels(N, k)
    X = [np.ones([m, n])*C[Ni] for Ni in range(N)]
    centroids = [np.ones([m, n]) * ki for ki in range(k)]
    expected = (C, X)
    actual = local.compute_clustering(X, centroids)
    assert all([np.array_equal(a, e) for a, e in zip(expected, actual)]))
"""
