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
import logging

import dkmeans.local_computations as local
import dkmeans.remote_computations as remote
from dkmeans.data import split_over_nodes, get_dataset, get_data_dims


DEFAULT_optimization = 'lloyd'
DEFAULT_s = 2
DEFAULT_epsilon = 0.00001
DEFAULT_shuffle = True
DEFAULT_lr = 0.001
DEFAULT_verbose = True


logger = logging.getLogger('dkmeans')
logger.setLevel(logging.INFO)


def main(X, k, optimization=DEFAULT_optimization, s=DEFAULT_s,
         epsilon=DEFAULT_epsilon, shuffle=DEFAULT_shuffle, lr=DEFAULT_lr,
         verbose=DEFAULT_verbose):
    m, n = get_data_dims(X)
    nodes, inds = split_over_nodes(X, s, shuffle=shuffle)
    X = [X[i] for i in inds]  # Reshuffle x to match the random
    tracked_delta = []
    num_iter = 0
    not_converged = True

    # Have each site compute k initial clusters locally
    local_centroids = [cent for node in nodes for cent in
                       local.initialize_own_centroids(node, k)]
    # and select k random clusters from the s*k pool
    np.random.shuffle(local_centroids)
    remote_centroids = local_centroids[:k]

    # Remote Optimization Loop
    while not_converged:
        cluster_labels = [None for j in range(s)]  # the clusterings
        local_optimizer = [None for j in range(s)]  # the optimization entity

        # Local computation loop
        for i, node in enumerate(nodes):
            # Each site compute local clusters
            cluster_labels[i] = \
                        local.compute_clustering(node, remote_centroids)
            if optimization == 'lloyd':
                # Lloyd has sites compute means locally
                local_optimizer[i] = local.compute_mean(node,
                                                        cluster_labels[i],
                                                        k)
            elif optimization == 'gradient':
                # Gradient descent has sites compute gradients locally
                local_optimizer[i] = \
                    local.compute_gradient(node, cluster_labels[i],
                                           remote_centroids, lr)
        # End of Local Computations

        # Both objects can be aggregated by taking a sum
        remote_optimizer = remote.aggregate_sum(local_optimizer)
        if optimization == 'lloyd':
            # and for the mean, we need to further divide the number of sites
            remote_optimizer = [r / s for r in remote_optimizer]

            # Then, update centroids as corresponding to the local mean
            previous = remote_centroids[:]
            remote_centroids = remote_optimizer[:]

        elif optimization == 'gradient':
            # Then, update centroids according to one step of gradient descent
            [remote_centroids, previous] = \
                local.gradient_step(remote_optimizer, remote_centroids)

        # Check the stopping condition "locally" at the aggregator
        # - returns false if converged
        remote_check, delta = local.check_stopping(remote_centroids,
                                                   previous, epsilon)
        if verbose:
            logger.info("Multi-Shot %s ; iter : %d delta : %f"
                        % (optimization, num_iter, delta))
        not_converged = remote_check
        tracked_delta.append(delta)
        num_iter += 1

    # Compute the final clustering "locally" at the aggregator
    cluster_labels = [clusters for node in nodes for clusters in
                      local.compute_clustering(node, remote_centroids)]
    return {'centroids': remote_centroids, 'cluster_labels': cluster_labels,
            'X': X, 'delta': tracked_delta, 'num_iter': i,
            'name': 'multishot_%s' % optimization}


if __name__ == '__main__':
    w = main(get_dataset(100, theta=[[-1, 0.1], [1, 0.1]], m=1, n=2), 2)
