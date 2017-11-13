"""
Compute Single-Shot K-Means with Expectation Maximization

Algorithm Flow -
    1: On each site, initialize Random Centroids 
    2: On each site, compute a clustering C with k-many clusters
    3: On each site, compute a local mean for each cluster in C
    4: On each site, recompute centroids as equal to local means
    5: On each site,
        if change in centroids below some epsilon, STOP, report STOPPED
        else GOTO step 3
    6: On each site, broadcast local centroids to aggregator
    7: On the aggregator, compute merging of clusters according to 
        least merging error (e.g. smallest distance betweeen centroids)
    8: Broadcast merged centroids to all sites
"""

import numpy as np
from dkmeans_util import (plot_clustering, simulated_gaussian_cluster,
                          random_split_over_nodes,
                          plot_clustering_node_subplots,
                          save_frames_to_gif)
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn import datasets
from dfnc_load_data import window_all_tc, load_exemplar, load_real_exemplar


def dkm_ss_em_local_initialize_centroids(n, m, k):
    w = [np.random.randn(m, n) for i in range(k)]
    return w


def dkm_ss_em_local_initialize_own_centroids(local_D, k):
    ii = np.random.choice(len(local_D), k)
    w = []
    for i in ii:
        w += [local_D[i]]
    return w


def dkm_ss_em_local_update_centroids(local_C, local_D, w, k):
    wo = [np.array(wi) for wi in w]
    M = [[] for i in range(k)]
    for i in range(len(local_C)):
        M[local_C[i]] += [local_D[i]]
    for i in range(k):
        if len(M[i]) is 0:
            M[i] = [np.zeros(wo[0].shape)]
        w[i] = np.mean(M[i], 0)
    return w, wo


def dkm_ss_em_local_check_stopping(w, wo, epsilon):
    delta = np.sum([cdist(np.matrix(w[i]),
                    np.matrix(wo[i])) for i in range(len(w))])
    # print("Delta", delta)
    result = delta > epsilon
    return result, delta


def dkm_ss_em_local_compute_clustering(local_D, remote_w):
    C = []
    Z = []
    for d in local_D:
        ld = [cdist(np.matrix(d), np.matrix(w)) for w in remote_w]
        min_i = ld.index(min(ld))
        C += [min_i]
        Z += [remote_w[min_i]]
    return C, Z


def dkm_ss_em_remote_aggregate_clusters(W, k, s, m, n):
    ks = k*s
    Wl = []
    Wt = []
    for w in W:
        Wl += [wi.reshape(1, m*n) for wi in w]
        Wt += [wi for wi in w]
    Wf = np.matrix(np.vstack(Wl))
    while ks > k:
        d = (squareform(pdist(Wf)))
        np.fill_diagonal(d, np.Inf)
        a = np.where(d == d.min())[0]
        P = [Wf[a[0]], Wf[a[1]]]
        Wf = np.delete(Wf, a[0], 0)
        Wf = np.delete(Wf, a[1]-1, 0)
        del Wt[a[0]]
        del Wt[a[1]-1]
        Wnew = (P[0] + P[1])/2.0
        Wf = np.vstack([Wf, Wnew])
        Wt += [Wnew.reshape(m, n)]
        ks -= 1
    return Wt


def main(X, k, s=2, ep=0.00001):
    try:
        [m, n] = X[0].shape
    except ValueError:
        [m, n] = 1, X[0].size
    nodes, inds = random_split_over_nodes(X, s)
    # print(inds)
    D = []
    for node in nodes:
        D += node
    i = 0
    b = True
    W = [None for j in range(s)]
    Del = []
    for si in range(s):
        W[si] = dkm_ss_em_local_initialize_own_centroids(nodes[si], k)
        # W[si] = dkm_ss_local_initialize_centroids(n, m, k)
    while b:
        b = False
        C = [None for j in range(s)]
        Deli = []
        Z = [None for j in range(s)]
        # print("Iteration %d" % i)
        for si in range(s):
            node = nodes[si]
            Ci, Zi = dkm_ss_em_local_compute_clustering(node, W[si])
            [w, w0] = dkm_ss_em_local_update_centroids(Ci, node, W[si], k)
            C[si] = Ci
            Z[si] = Zi
            W[si] = w
            dis, bi = dkm_ss_em_local_check_stopping(w, w0, ep)
            Deli += [dis]
            b = b or bi
        i += 1
        Del += [Deli]
    w = dkm_ss_em_remote_aggregate_clusters(W, k, s, m, n)
    w = [np.array(wi) for wi in w]
    C = []
    for si in range(s):
        node = nodes[si]
        Ci, Zi = dkm_ss_em_local_compute_clustering(node, w)
        # print(len(Ci))
        C += Ci
    # print("Converged")
    return {'w': w, 'C': C, 'X': D, 'delta': Del, 'iter': i, 'name':'ss_em'}


if __name__ == '__main__':
    w = main()
