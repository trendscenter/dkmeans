"""
Compute Single-Shot K-Means with Gradient Descent

Algorithm Flow -
    1: On each site, initialize Random Centroids 
    2: On each site, compute a clustering C with k-many clusters
    3: On each site, compute a local gradient for each cluster in C
    4: On each site, update centroids via gradient descent
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
from scipy.spatial.distance import pdist, squareform
from sklearn import datasets
from dfnc_load_data import window_all_tc, load_exemplar, load_real_exemplar

def dkm_ss_local_initialize_centroids(n, m, k):
    w = [np.random.randn(m, n) for i in range(k)]
    return w


def dkm_ss_local_initialize_own_centroids(local_D, k):
    ii = np.random.choice(len(local_D), k)
    w = []
    for i in ii:
        w += [local_D[i]]
    return w


def dkm_ss_local_update_centroids(local_G, w):
    wo = [np.array(wi) for wi in w]
    for k in range(len(w)):
        w[k] += np.reshape(local_G[:, :, k], w[k].shape)
    return w, wo


def dkm_ss_local_check_stopping(w, wo, epsilon):
    delta = np.sum([abs(w[i] - wo[i]) for i in range(len(w))])
    # print("Delta", delta)
    result = delta > epsilon
    return result, delta


def dkm_ss_local_compute_clustering(local_D, remote_w):
    C = []
    Z = []
    for d in local_D:
        ld = [np.linalg.norm((abs(d-w))) for w in remote_w]
        min_i = ld.index(min(ld))
        C += [min_i]
        Z += [remote_w[min_i]]
    return C, Z


def dkm_ss_local_compute_gradient(local_D, local_C, local_Z,
                                  remote_w, remote_e, m, n, k):
    local_G = np.zeros([m, n, k])
    for i in range(len(local_D)):
        d = local_D[i]
        c = local_C[i]
        z = local_Z[i]
        local_G[:, :, c] += remote_e*(d - z)
    return local_G


def dkm_ss_remote_aggregate_clusters(W, k, s, m, n):
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


def main(X, k, s=2, ep=0.00001, e=0.0001, N=2000):
    try:
        [m, n] = X[0].shape
    except ValueError:
        [m, n] = 1, X[0].size
    nodes, inds = random_split_over_nodes(X, s)
    D = []
    for node in nodes:
        D += node
    i = 0
    b = True
    W = [None for j in range(s)]
    for si in range(s):
        W[si] = dkm_ss_local_initialize_own_centroids(nodes[si], k)
        # W[si] = dkm_ss_local_initialize_centroids(n, m, k)
    Del = []
    while b:
        b = False
        Deli = [None for j in range(s)]
        C = [None for j in range(s)]
        Z = [None for j in range(s)]
        # print("Iteration %d" % i)
        for si in range(s):
            node = nodes[si]
            Ci, Zi = dkm_ss_local_compute_clustering(node, W[si])
            liG = dkm_ss_local_compute_gradient(node, Ci, Zi, W[si],
                                                e, m, n, k)
            [w, w0] = dkm_ss_local_update_centroids(liG, W[si])
            C[si] = Ci
            Z[si] = Zi
            W[si] = w
            bi, Deli[si] = dkm_ss_local_check_stopping(w, w0, ep)
            b = b or bi
        i += 1
        Del += [Deli]
        if i > N:
            break
    w = dkm_ss_remote_aggregate_clusters(W, k, s, m, n)
    w = [np.array(wi) for wi in w]
    C = []
    for si in range(s):
        node = nodes[si]
        Ci, Zi = dkm_ss_local_compute_clustering(node, w)
        C += Ci
    # print("Converged")
    return {'w': w, 'C': C, 'X': D, 'delta': Del, 'iter': i, 'name': 'ss_gd'}


if __name__ == '__main__':
    w = main()
