# -*- coding: utf-8 -*-
"""

Perform pooled k-means clustering using the sklearn library
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
