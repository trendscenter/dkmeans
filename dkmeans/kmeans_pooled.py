# -*- coding: utf-8 -*-
"""

Perform pooled k-means clustering using the sklearn library
"""
import numpy as np
from sklearn.cluster import KMeans


DEFAULT_steps = np.Inf
DEFAULT_sites = 0
DEFAULT_verbose = False


def main(X, k, steps=DEFAULT_steps, s=DEFAULT_sites, verbose=DEFAULT_verbose):
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
    return {'centroids': w, 'cluster_labels': labels, 'X': X,
            'num_iter': kmeans_model.n_iter_, 'name': 'pooled'}


if __name__ == '__main__':
    w = main()
