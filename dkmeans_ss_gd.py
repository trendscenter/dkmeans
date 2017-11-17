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

# Module Imports
import numpy as np


from dkmeans_util import random_split_over_nodes
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.spatial.distance import cdist


def dkm_ss_gd_local_gaussian_centroids(n, m, k):
    """ Random gaussian centroids """
    w = [np.random.randn(m, n) for i in range(k)]
    return w


def dkm_ss_gd_local_initialize_own_centroids(local_D, k):
    """ Random choice of k centroids from own data """
    ii = np.random.choice(len(local_D), k)
    w = []
    for i in ii:
        w += [local_D[i]]
    return w


def dkm_ss_gd_local_update_centroids(local_G, w):
    """ Gradient descent update on local site """
    wo = [np.array(wi) for wi in w]
    for k in range(len(w)):
        w[k] += np.reshape(local_G[:, :, k], w[k].shape)
    return w, wo


def dkm_ss_gd_local_check_stopping(w, wo, epsilon):
    """ Local sites check stopping condition"""
    delta = np.sum([cdist(w[i], wo[i]) for i in range(len(w))])
    result = delta > epsilon
    return result, delta


def dkm_ss_gd_local_compute_clustering(local_D, remote_w):
    """ Compute local clustering by nearest centroid """
    C = []
    Z = []
    for d in local_D:
        #  !!! a numpy shaping hack which I only need in this file...
        #  d for some reason comes in as a 1-d vector, which sklear
        #  doesn't like...
        ld = [cdist(d.reshape(w.shape), w) for w in remote_w]
        min_i = ld.index(np.min(ld))
        C += [min_i]
        Z += [remote_w[min_i]]
    return C, Z


def dkm_ss_gd_local_compute_gradient(local_D, local_C, local_Z,
                                     remote_w, remote_e, m, n, k):
    """ Compute local gradient
        Local D is local data
        Local C is local clustering
        Local Z is the centroid associated with the cluster given in Local C
            TODO: Can we compute these distances without passing centroids?
    """
    local_G = np.zeros([m, n, k])
    for i in range(len(local_D)):
        d = local_D[i]
        c = local_C[i]
        z = local_Z[i]
        local_G[:, :, c] += remote_e*(d - z)
    return local_G


def dkm_ss_gd_remote_aggregate_clusters(W, k, s, m, n):
    """
        Naive cluster aggregation by closest-centroid merging strategy
    """
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
    """
        Run the distributed framework in simulation.
        parameters:
            k - number of centroids
            ep - epsilon tolerance
            e - learning rate
            N - number of simulated samples
    """
    try:
        [m, n] = X[0].shape
    except ValueError:
        [m, n] = 1, X[0].size
    nodes, inds = random_split_over_nodes(X, s)
    D = []
    for node in nodes:
        D += node
    i = 0
    not_converged = True
    W = [None for j in range(s)]

    # All sites initialize their own centroids
    for si in range(s):
        W[si] = dkm_ss_gd_local_initialize_own_centroids(nodes[si], k)
        W[si] = [w.reshape([m, n]) for w in W[si]]
    Delta = []
    # Training loop
    while not_converged:
        not_converged = False
        Deli = [None for j in range(s)]
        C = [None for j in range(s)]  # the clusterings
        Z = [None for j in range(s)]  # centroid associated with those clusters

        # At each site
        for si in range(s):
            node = nodes[si]
            # Update Local clustering
            Ci, Zi = dkm_ss_gd_local_compute_clustering(node, W[si])

            # Compute the Gradients
            liG = dkm_ss_gd_local_compute_gradient(node, Ci, Zi, W[si],
                                                e, m, n, k)

            # Update centroids
            [w, w0] = dkm_ss_gd_local_update_centroids(liG, W[si])

            C[si] = Ci
            Z[si] = Zi
            W[si] = w

            # Check Stopping Conditions Locally
            local_check, Deli[si] = dkm_ss_gd_local_check_stopping(w, w0, ep)

            # But only stop when all sites return False
            not_converged = not_converged or local_check
        i += 1
        Delta.extend(Deli)
        if i > N:
            # Number of iterations exceeded
            break

    # Perform one-time cluster merging
    w = dkm_ss_gd_remote_aggregate_clusters(W, k, s, m, n)
    w = [np.array(wi) for wi in w]
    C = []
    for si in range(s):
        node = nodes[si]
        Ci, Zi = dkm_ss_gd_local_compute_clustering(node, w)
        C += Ci

    return {'w': w, 'C': C, 'X': D, 'delta': Delta, 'iter': i, 'name': 'ss_gd_gd'}
