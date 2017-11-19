#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 19:52:34 2017

@author: bbradt
"""

import unittest
import numpy as np
from dkmeans_remote_computations import dkm_remote_aggregate_clusters
from dkmeans_remote_computations import dkm_remote_aggregate_sum
from dkmeans_remote_computations import dkm_remote_closest_centroids
from dkmeans_local_computations import dkm_local_compute_mean
from dkmeans_local_computations import dkm_local_mean_step
from dkmeans_local_computations import dkm_local_compute_clustering


DEFAULT_PARAMS = (2,   # m
                  2,   # n
                  2,   # k
                  3,   # s
                  11,  # N
                  )


def _k_cluster_labels(N, k):
    C = []
    for ki in range(k):
        C.extend([ki for i in range(int(N/k))])
    while len(C) < N:
        C.append(k - 1)
    return C


class TestRemoteComputations(unittest.TestCase):

    def test_dkm_remote_aggregate_clusters(self):
        m, n, k, s, _ = DEFAULT_PARAMS
        centroids = [[np.ones([m, n]) for ki in range(k)] for si in range(s)]
        expected = [np.ones([m, n]) for ki in range(k)]
        actual = dkm_remote_aggregate_clusters(centroids)
        self.assertTrue(all([np.array_equal(a, e)
                        for a, e in zip(expected, actual)]))

    def test_dkm_remote_aggregate_sum(self):
        objects = [[np.array([0]), np.array([1])],
                   [np.array([1]), np.array([0])]
                   ]
        expected = [np.array([1]), np.array([1])]
        actual = dkm_remote_aggregate_sum(objects)
        np.testing.assert_array_equal(expected, actual)

    def test_dkm_remote_closest_centroids(self):
        m, n, k, s, _ = DEFAULT_PARAMS
        centroids = [[np.ones([m, n])
                     for ki in range(k)] for si in range(s)]
        actual = dkm_remote_closest_centroids(centroids)
        expected = [(i, i+1) for i in range(0, k*s-1, 2)]
        if k*s % 2 != 0:
            expected.append((k*s-1, k*s-1))
        self.assertEqual(actual, expected)

    '''
class TestLocalComputations(unittest.TestCase):

    def test_dkm_local_compute_mean(self):
        m, n, k, s, N = DEFAULT_PARAMS
        X = [np.ones([m, n]) for Ni in range(N)]
        C = _k_cluster_labels(N, k)
        expected = [np.ones([m, n]) for ki in range(k)]
        actual = dkm_local_compute_mean(X, C, k)
        np.testing.assert_array_equal(expected, actual)

    def test_dkm_local_mean_step(self):
        m, n, k, s, N = DEFAULT_PARAMS
        C = _k_cluster_labels(N, k)
        X = [np.ones([m, n]) for Ni in range(N)]
        centroids = [[np.ones([m, n]) for ki in range(k)] for si in range(s)]
        local_means = dkm_local_compute_mean(X, C, k)
        expected = ([lmean for lmean in local_means], centroids)
        actual = dkm_local_mean_step(local_means, centroids)
        self.assertTrue(all([np.array_equal(a, e)
                        for a, e in zip(expected, actual)]))

    def test_dkm_local_compute_clustering(self):
        m, n, k, s, N = DEFAULT_PARAMS
        C = _k_cluster_labels(N, k)
        X = [np.ones([m, n])*C[Ni] for Ni in range(N)]
        centroids = [np.ones([m, n]) * ki for ki in range(k)]
        expected = (C, X)
        actual = dkm_local_compute_clustering(X, centroids)
        self.assertTrue(all([np.array_equal(a, e)
                        for a, e in zip(expected, actual)]))
            '''


if __name__ == '__main__':
    unittest.main()
