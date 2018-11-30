#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:40:31 2018

@author: bbaker
"""

import numpy as np
from scipy.spatial.distance import cdist


def average_distance(data, labels):
    clusters = [[], [], [], [], []]
    for x, label in zip(data, labels):
        clusters[label].append(x)
    centroids = [np.mean(cluster) for cluster in clusters]
    distances = [cdist(x, centroid, metric='correlation') for cluster, centroid
                 in zip(clusters, centroids) for x in cluster]
    return np.mean(distances)


def total_distance(data, labels):
    clusters = [[]*(max(labels)+1)]
    for x, label in zip(data, labels):
        clusters[label].append(x)
    centroids = [np.mean(cluster) for cluster in clusters]
    distances = [cdist(x, centroid, metric='correlation') for cluster, centroid
                 in zip(clusters, centroids) for x in cluster]
    return np.sum(distances)
