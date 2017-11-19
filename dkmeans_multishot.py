# -*- coding: utf-8 -*-
"""
Compute Multi-Shot K-Means with LLoyd's algorithm or Gradient Descent

Algorithm Flow -
    1: On the aggregator, initialize random Centroids
        (either entirely remotely computed or shared between local sites)
    2: Broadcast Centroids to all Sites
    3: On each site, compute a clustering C with k-many clusters
    [lloyd's algorithm]
        4: On each site, compute a local mean for each cluster in C
        5: On each site, broadcast local mean to the aggregator
        6: On the aggregator, compute the global means for each Cluster
        7: On the aggregator, recompute centroids as equal to global means
    [gradient descent]
        4: On each site, compute a local gradient for each cluster in C
        5: On each site, broadcast local gradient to the aggregator
        6: On the aggregator, compute the global gradients for each Cluster
        7: On the aggregator, update centroids according to gradient descent
    8: On the aggregator,
        if change in centroids below some epsilon, broadcast STOP
        else broadcast new centroids, GOTO step 3
"""

import numpy as np


from dkmeans_local_computations import dkm_local_initialize_own_centroids
from dkmeans_local_computations import dkm_local_compute_clustering
from dkmeans_local_computations import dkm_local_compute_mean
from dkmeans_local_computations import dkm_local_compute_gradient
from dkmeans_local_computations import dkm_local_check_stopping
from dkmeans_local_computations import dkm_local_gradient_step
from dkmeans_local_computations import dkm_local_mean_step
from dkmeans_remote_computations import dkm_remote_aggregate_sum
from dkmeans_remote_computations import dkm_remote_closest_centroids
from dkmeans_data import split_over_nodes, get_dataset, get_data_dims

from dkmeans_plot import plot_2d_clustering


def main(X, k, optimization='lloyd', s=2, epsilon=0.00001, shuffle=True,
         lr=0.001, verbose=True):
    m, n = get_data_dims(X)
    nodes, inds = split_over_nodes(X, s, shuffle=shuffle)
    X = [X[i] for i in inds]  # Reshuffle x to match the random
    tracked_delta = []
    num_iter = 0
    not_converged = True
    local_centroids = [cent for node in nodes for cent in
                       dkm_local_initialize_own_centroids(node, k)]
    np.random.shuffle(local_centroids)
    remote_centroids = local_centroids[:k]
    while not_converged:
        cluster_labels = [None for j in range(s)]  # the clusterings
        local_optimizer = [None for j in range(s)]
        for i, node in enumerate(nodes):
            cluster_labels[i] = \
                        dkm_local_compute_clustering(node, remote_centroids)
            if optimization == 'lloyd':
                local_optimizer[i] = dkm_local_compute_mean(node,
                                                            cluster_labels[i],
                                                            k)
            elif optimization == 'gradient':
                local_optimizer[i] = \
                    dkm_local_compute_gradient(node, cluster_labels[i],
                                               remote_centroids, lr)
        # The remote optimizer contains s * k many means
        remote_optimizer = dkm_remote_aggregate_sum(local_optimizer)
        if optimization == 'lloyd':
            remote_optimizer = [r / s for r in remote_optimizer]
            [remote_centroids, previous] = \
                dkm_local_mean_step(remote_optimizer,
                                    remote_centroids)
        elif optimization == 'gradient':
            [remote_centroids, previous] = \
                dkm_local_gradient_step(remote_optimizer, remote_centroids)

        remote_check, delta = dkm_local_check_stopping(remote_centroids,
                                                       previous, epsilon)
        if verbose:
            print("Multi-Shot %s ; iter : %d delta : %f"
                  % (optimization, num_iter, delta))
        not_converged = remote_check
        tracked_delta.append(delta)
        num_iter += 1

    cluster_labels = [clusters for node in nodes for clusters in
                      dkm_local_compute_clustering(node, remote_centroids)]
    return {'centroids': remote_centroids, 'cluster_labels': cluster_labels,
            'X': X, 'delta': tracked_delta, 'num_iter': i,
            'name': 'multishot_%s' % optimization}


if __name__ == '__main__':
    w = main(get_dataset(100, theta=[[-1, 0.1], [1, 0.1]], m=1, n=2), 2)
