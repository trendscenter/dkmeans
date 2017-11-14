"""
Compute Multi-Shot K-Means with LLoyd's algorithm

Algorithm Flow -
    1: On the aggregator, initialize random Centroids
        (either entirely remotely computed or shared between local sites)
    2: Broadcast Centroids to all Sites
    3: On each site, compute a clustering C with k-many clusters
    4: On each site, compute a local mean for each cluster in C
    5: On each site, broadcast local mean to the aggregator
    6: On the aggregator, compute the global means for each Cluster
    7: On the aggregator, recompute centroids as equal to global means
    8: On the aggregator,
        if change in centroids below some epsilon, broadcast STOP
        else broadcast new centroids, GOTO step 3
"""

import numpy as np


from dkmeans_util import random_split_over_nodes
from scipy.spatial.distance import cdist


def dkm_ms_ll_remote_gaussian_centroids(n, m, k, mu=0.0, sigsq=1.0):
    """
    Initialize centroids using random gaussian distribution.
    Requires    - global size of data instances m x n
                - mu and sigma for determining shape of distribution
    Returns k centroids given as m x n Numpy arrays.
    """
    w = [(sigsq*np.random.randn(m, n) + mu).flatten() for i in range(k)]
    return w


def dkm_ms_ll_local_initialize_own_centroids(local_D, k):
    ii = np.random.choice(len(local_D), k)
    w = []
    for i in ii:
        w += [local_D[i]]
    return w


def dkm_ms_ll_local_compute_mean(local_D, local_C, k):
    """
    Compute the local mean, which is broadcast back to the agg
    """
    M = [[] for i in range(k)]
    for i in range(len(local_C)):
        M[local_C[i]] += [local_D[i]]
    for i in range(k):
        M[i] = np.mean(M[i], 0).flatten()
    return M


def dkm_ms_ll_remote_update_centroids(remote_M, w, k):
    """
    Centroids updated as equal to the remote Means
    """
    wo = [np.array(wi) for wi in w]
    for i in range(k):
        w[i] = np.mean(remote_M[i], 0).flatten()
    return w, wo


def dkm_ms_ll_remote_check_stopping(w, wo, epsilon):
    """
    Stopping condition is distance below some epsilon
    """
    delta = np.sum([abs(w[i] - wo[i]) for i in range(len(w))])
    # print("Delta", delta)
    result = delta > epsilon
    return result, delta


def dkm_ms_ll_local_compute_clustering(local_D, remote_w):
    C = []
    Z = []
    for d in local_D:
        ld = [cdist(np.matrix(d), np.matrix(w)) for w in remote_w]
        min_i = ld.index(np.min(ld))
        C += [min_i]
        Z += [remote_w[min_i]]
    return C, Z


def main(X, k, s=2, ep=0.00001):
    try:
        [m, n] = X[0].shape
    except ValueError:
        [m, n] = 1, X[0].size  # in case X is 0-dim
    nodes, inds = random_split_over_nodes(X, s)
    w = []

    # At each sites, initialize centroids
    for si in range(s):
        # Have each node choose K centroids, for a total of s*k, randomly
        # keep k many of them
        wi = dkm_ms_ll_local_initialize_own_centroids(nodes[si], k)
        w += wi
    np.random.shuffle(w)
    w = w[:k]

    not_converged = True
    i = 0  # number of iterations
    D = []  # global data
    Delta = []  # Delta
    for node in nodes:
        D += node
    while not_converged:
        lM = [[] for i in range(k)]
        C = []
        Cl = []
        Z = []
        for node in nodes:
            Ci, Zi = dkm_ms_ll_local_compute_clustering(node, w)
            Cl += [Ci]
            liM = dkm_ms_ll_local_compute_mean(node, Ci, k)
            lM = [lM[i] + [liM[i]] for i in range(k)]
            C += Ci
            Z += Zi
        [w, w0] = dkm_ms_ll_remote_update_centroids(lM, w, k)
        b, delta = dkm_ms_ll_remote_check_stopping(w, w0, ep)
        Delta += [delta]
        i += 1
    return {'w': w, 'C': C, 'X': D, 'delta': Delta, 'iter': i, 'name': 'ms_ll'}
