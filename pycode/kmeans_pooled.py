#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:19:54 2017

@author: bbaker
"""
import numpy as np
from sklearn.cluster import KMeans


def main(X, k, s, steps=np.Inf):
    try:
        [m, n] = X[0].shape
    except ValueError:
        [m, n] = 1, X[0].size
    X = np.array([x.flatten() for x in X])
    kmeans_model = KMeans(n_clusters=k,
                          random_state=1, init='random', n_init=10).fit(X)
    w = kmeans_model.cluster_centers_
    labels = kmeans_model.labels_
    w = [r.reshape([m, n]) for r in list(w)]
    return {'w': w, 'C': labels, 'X': X, 'iter': kmeans_model.n_iter_,
            'name': 'pooled'}

if __name__ == '__main__':
    w = main()
