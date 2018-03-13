#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Common functions for remote node operations, i.e. funcitons which involve
    data from multiple local nodes

"""
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


def closest_centroids(local_centroids):
    """
        Match local_centroids to neighbors with a **greedy** strategy,
        iteratively picking the next smallest merging distance

        Input:  local_centroids - local_centroids from local sites
                         a 2d list of s sites with k many m x n local_centroids
                Supposing s many sites with k centroids each,
                there are s * k initial centroids in the input

        Output: A list of tuples where each index in the tuple represents a
                pair of matching centroids. There will never be more than
                floor(s*k/2) greedy matches.

        TODO: Implement a non-greedy version, minimizing total merging
                distance
    """
    flat_centroids = np.matrix(
             np.vstack([np.vstack([wi.reshape(1, np.prod(wi.shape))
                                   for wi in w])
                       for w in local_centroids]))
    unions = []
    distances = squareform(pdist(flat_centroids))
    # fill the lower triangular with inf to make the relation asymmetric
    distances[np.tril_indices_from(distances)] = np.Inf
    while np.min(distances) != np.Inf:
        mincoords = np.where(distances == np.min(distances))
        i, j = mincoords[0][0], mincoords[1][0]  # if not unique, pick first
        unions.append((i, j))
        distances[i, :] = np.Inf
        distances[:, i] = np.Inf
        distances[j, :] = np.Inf
        distances[:, j] = np.Inf
    if len(flat_centroids) % 2 != 0:
        missing = set(range(len(flat_centroids))) - set([ind for u in unions
                                                         for ind in u])
        [unions.append((m, m)) for m in missing]
    return(unions)


def aggregate_clusters(local_centroids):
    """
        Cluster merging using closest centroid computations

        Input: local_centroids - local_centroids from local sites
                         a 2d list of s sites with k many m x n local_centroids

        Output: a list of updated centroids, the mid-point between the matching
                centroids, returned from closest_centroids.
                Supposing s*k many centroids initially, floor(s*k/2) centroids
                will be output, with an additional centroid if s and k are
                both odd
    """
    closest_indices = closest_centroids(local_centroids)
    local_centroids = [cent for sitecents in local_centroids
                       for cent in sitecents]
    return [np.sum([local_centroids[mincoords[0]],
                    local_centroids[mincoords[1]]], 0) / 2.0
            for mincoords in closest_indices]


def aggregate_sum(local_objects):
    return list(np.sum(local_objects, 0))
