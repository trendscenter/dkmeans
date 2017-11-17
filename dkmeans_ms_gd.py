"""
Compute Multi-Shot K-Means with Gradient-Descent

Algorithm Flow -
    1: On the aggregator, initialize random Centroids 
        (either entirely remotely computed or shared between local sites)
    2: Broadcast Centroids to all Sites
    3: On each site, compute a clustering C with k-many clusters
    4: On each site, compute a local gradient for each cluster in C
    5: On each site, broadcast local gradient to the aggregator
    6: On the aggregator, compute the global gradients for each Cluster
    7: On the aggregator, update centroids according to gradient descent
    8: On the aggregator,
        if change in centroids below some epsilon, broadcast STOP
        else broadcast new centroids, GOTO step 3

"""

import numpy as np
from dkmeans_util import (plot_clustering, simulated_gaussian_cluster,
                          random_split_over_nodes,
                          plot_clustering_node_subplots,
                          save_frames_to_gif)
from scipy.spatial.distance import cdist


def dkm_ms_remote_initialize_centroids(n, m, k):
    """ Random gaussian centroids """
    w = [np.random.randn(m, n).flatten() for i in range(k)]
    return w


def dkm_ms_local_initialize_own_centroids(local_D, k):
    """ Random local centroids """
    ii = np.random.choice(len(local_D), k)
    w = []
    for i in ii:
        w += [local_D[i]]
    return w


def dkm_ms_remote_aggregate_gradient(local_G, m, n, k):
    """ Aggregate global gradient as sum of local gradients """
    remote_G = np.zeros([m, n, k])
    for lG in local_G:
        remote_G += lG
    return remote_G


def dkm_ms_remote_update_centroids(remote_G, w):
    """ update centroids on aggregator only, then broadcast """
    wo = [np.array(wi) for wi in w]
    for k in range(len(w)):
        w[k] += remote_G[:, :, k].reshape(w[k].shape)
    return w, wo


def dkm_ms_remote_check_stopping(w, wo, epsilon):
    """ Check stopping condition on the aggregator """
    delta = np.sum([cdist(w[i], wo[i]) for i in range(len(w))])
    result = delta > epsilon
    return result, delta


def dkm_ms_local_compute_clustering(local_D, remote_w):
    C = []
    Z = []
    for d in local_D:
        ld = [cdist(np.matrix(d), np.matrix(w)) for w in remote_w]
        min_i = ld.index(np.min(ld))
        C += [min_i]
        Z += [remote_w[min_i]]
    return C, Z


def dkm_ms_local_compute_gradient(local_D, local_C, local_Z,
                                  remote_w, remote_e, m, n, k):
    local_G = np.zeros([m, n, k])
    for i in range(len(local_D)):
        d = local_D[i]
        c = local_C[i]
        z = local_Z[i]
        local_G[:, :, c] += remote_e*(d - z)
    return local_G


def main(X, k, s=2, ep=0.00001, e=0.0001, N=2000):
    try:
        [m, n] = X[0].shape
    except ValueError:
        [m, n] = 1, X[0].size
    nodes, inds = random_split_over_nodes(X, s)
    w = []
    # At each site initialize K centroids
    for si in range(s):
        wi = dkm_ms_local_initialize_own_centroids(nodes[si], k)
        w.extend(wi)
    # Then randomly select K from those k*s centroids
    np.random.shuffle(w)
    w = w[:k]
    w = [wi.reshape([m, n]) for wi in w]
    not_converged = True  # convergence
    i = 0  # iterations
    D = []
    Delta = []
    for node in nodes:
        D += node
    while not_converged:
        not_converged = False
        lG = []
        C = []
        Cl = []
        Z = []
        for node in nodes:
            Ci, Zi = dkm_ms_local_compute_clustering(node, w)
            Cl += [Ci]
            liG = dkm_ms_local_compute_gradient(node, Ci, Zi, w, e, m, n, k)
            lG += [liG]
            C += Ci
            Z += Zi
        # Remote steps - Aggregate Gradient, Update Centroids, Check Stopping
        G = dkm_ms_remote_aggregate_gradient(lG, m, n, k)
        [w, w0] = dkm_ms_remote_update_centroids(G, w)
        remote_check, delta = dkm_ms_remote_check_stopping(w, w0, ep)
        not_converged = not_converged or remote_check
        i += 1
        Delta += [delta]
        if i > N:
            break
    return {'w': w, 'C': C, 'X': D, 'delta': Delta, 'iter': i, 'name': 'ms_gd'}
