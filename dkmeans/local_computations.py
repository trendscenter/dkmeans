#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Common functions for local node operations

Common Input Specifications:
    local_X - local data at site, list of N many m x n numpy arrays
    local_cluster_labels - local data cluster assignments,
                list of N many integers in range(k)
    local_centroids - list of k many m x n centroids
    k - number of clusters (integer)
"""
import numpy as np
from scipy.spatial.distance import cdist
from dkmeans.data import get_data_dims


def compute_mean(local_X, local_cluster_labels, k):
    """
        Compute the local mean, which is broadcast back to the aggregator

        Input: local_X, local_cluster_labels, k as above

        Output: list of k many local mean matrices, shape m x n
    """
    m, n = get_data_dims(local_X)
    npinf = np.zeros([m, n])
    local_means = [[]]*k
    for label, x in zip(local_cluster_labels, local_X):
        local_means[label] += [x]

    #  Return the origin if no clusters have been assigned to cluster k
    #  !!! is this the way to handle this?
    return [np.mean(lmean, 0) if lmean else npinf for lmean in local_means]


def gradient_step(local_gradients, local_centroids):
    """
        Gradient descent update on local site

        Input: local_gradients - list of k many local gradients

        Output: updated local centroids, previous centroids from last iteration
    """
    previous = local_centroids[:]
    local_centroids = [wk + local_gradients[k]
                       for k, wk in enumerate(local_centroids)]
    return local_centroids, previous


def initialize_own_centroids(local_X, k):
    """
        Random choice of k centroids from own data

        Input: local_X, k as above

        Output: list of k many points selected from local_X
    """
    return [local_X[i] for i in np.random.choice(len(local_X), k)]


def check_stopping(local_centroids, previous_centroids, epsilon):
    """
        Check if centroids have changed beyond some epsilon tolerance

        Input: local_centroids as above
                previous_centroids, the centroids from the prior iteration
                epsilon - the tolerance threshold (float)

        Output: True if delta is above the threshold, else False
    """
    m, n = get_data_dims(local_centroids)
    flat_centroids = [np.matrix(w.reshape(1, m*n)) for w in local_centroids]
    flat_previous = [np.matrix(w.reshape(1, m*n)) for w in previous_centroids]
    # delta is the change in centroids, computed by distance metric
    delta = np.sum([cdist(w, flat_previous[k])
                    for k, w in enumerate(flat_centroids)])
    return delta > epsilon, delta


def compute_clustering(local_X, local_centroids):
    """
        Compute local clustering by associating each data instance with the
        nearest centroid

        Input: local_X, centroids as above

        Output: cluster_labels- a list of N many integers,
                                    the labels for each instance
    """
    cluster_labels = []
    m, n = get_data_dims(local_X)
    X_flat = [np.matrix(x.reshape(1, m*n)) for x in local_X]
    w_flat = [np.matrix(w.reshape(1, m*n)) for w in local_centroids]
    for x in X_flat:
        distances = [cdist(x, w) for w in w_flat]
        min_index = distances.index(np.min(distances))
        cluster_labels.append(min_index)
    return cluster_labels


def compute_gradient(local_X, local_cluster_labels, local_centroids, lr):
    """
        Compute local gradient
        Input:  local_X, local_cluster_labels, local_centroids as above
                lr - the learning rate (float)

        Output: local_grad - local gradients as list of k many gradients
    """
    m, n = get_data_dims(local_X)
    local_grad = [np.zeros([m, n])
                  for i, e in enumerate(local_centroids)]
    for x, i in zip(local_X, local_cluster_labels):
        local_grad[i] += lr*(x - local_centroids[i])
    return local_grad
