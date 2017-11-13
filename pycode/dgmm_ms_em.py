"""
Compute Multi-Shot Gaussian Mixture Model with Expectation Maximization

Algorithm Flow -
    1: On the aggregator, initialize random normal distributions, Theta
    2: Broadcast Theta to all sites
    3: all sites, compute weights for each cluster according to local data
    4: all sites, compute partial Nk 
    5: all sites, broadcast partial Nk and weights to aggregator
    6: Aggregator, compute mu for each cluster k, broadcast to sites
    7: All sites, compute partial sigma_k pass to aggregator
    8: Aggregator, compute sigma_k, broadcast to all sites
    9: All sites, locally compute partial log-likelihood
    10: Aggregator check change in log-likelihood
            if below epsilon, broadcast STOP
            else GOTO 3
"""

import numpy as np
from dkmeans_util import (plot_clustering, simulated_gaussian_cluster,
                          random_split_over_nodes,
                          plot_clustering_node_subplots,
                          save_frames_to_gif)


def dgmm_ms_em_remote_initialize_random_gaussian(N,k):
    muk = [np.random.randn(1,1).flatten() for i in range(k)]
    sigmak = [np.random.randn(N,N) for i in range(k)]
    for i in range(k):
        sigmak[i] = np.dot(sigmak[i],sigmak[i].T)
    return muk, sigmak


def dgmm_ms_em_local_gaussian(remote_mu_k, remote_sigma_k, local_x, k):
    pk = [None for i in range(k)]
    for i in range(k):
        diff = local_x - remote_mu_k[i]
        num = np.exp((-1/2)*np.dot(diff.T,np.linalg.inv(remote_sigma_k[i])).dot(diff))
        denom = 2*np.pi*np.power(np.linalg.det(remote_sigma_k[i]),1/2)
        pk[i] = num/denom
        print(pk[i])     
    return pk


def dgmm_ms_em_update_local_weights(local_pk, local_ak, k):
    local_wk = [None for i in range(k)]
    denom = np.dot(local_pk, local_ak)
    for i in range(k):
        local_wk[i] = (local_pk[i] * local_ak[i])/denom
    return local_wk


def dgmm_ms_em_compute_local_Nk(local_wk, k):
    local_Nk = [None for i in range(k)]
    for i in range(k):
        local_Nk[i] = np.sum(local_wk[i])
    return local_Nk


def dgmm_ms_em_compute_remote_Nk(local_Nk, k):
    remote_Nk = [np.sum(local_Nk[i]) for i in range(k)]
    return remote_Nk


def dgmm_ms_em_compute_remote_Mk(local_Mk, k):
    remote_Nk = [np.sum(local_Mk[i]) for i in range(k)]
    return remote_Nk


def dgmm_ms_em_compute_local_Mk(local_wk, local_x):
    local_Mk = [None for i in range(k)]
    for i in range(k):
        local_Mk[i] = np.dot(local_wk[i], local_x)
    return local_Mk


def dgmm_ms_em_compute_remote_muK(remote_Nk, remote_Mk, k):
    remote_muK = [None for i in range(k)]
    for i in range(k):
        remote_muK = remote_Mk[i]/remote_Nk[i]
    return remote_muK


def dgmm_ms_em_compute_local_sigmaK(local_x, remote_muk,
                                    local_wk, k):
    local_sigmaK = [None for i in range(k)]
    for i in range(k):
        diff = local_x - remote_muk[i]
        C = np.dot(local_wk[i], diff.T*diff)
        local_sigmaK[i] = C
    return local_sigmaK


def dgmm_ms_em_compute_remote_sigmaK(local_sigmaK, remote_Nk, k):
    remote_sigmaK = [None for i in range(k)]
    for i in range(k):
        remote_sigmaK[i] = np.sum(local_sigmaK[i])/remote_Nk[i]
    return remote_sigmaK


def dgmm_ms_em_remote_update_alphak(remote_Nk, N, k):
    return [remote_Nk[i]/N for i in range(k)]


def dgmm_ms_em_compute_local_llog(local_p, remote_ak, k):
    local_llog = np.zeros(local_p[0].shape)
    for i in range(len(local_p)):
        local_llog += np.log(local_p)
    return local_llog


def dgmm_ms_em_compute_remote_llog(local_llog):
    return np.sum(local_llog)


def dkm_ms_em_remote_check_stopping(w, wo, epsilon):
    """
    Stopping condition is distance below some epsilon
    """
    delta = np.sum([abs(w[i] - wo[i]) for i in range(len(w))])
    # print("Delta", delta)
    result = delta > epsilon
    return result


if __name__ == '__main__':

    N = 100  # number of samples
    m = 1  # m x n data-points
    n = 2  # m x n data-points
    k = 3  # number of esimated clusters
    s = 2
    e = 0.01  # learning rate
    ep = 0.00001  # epsilon tolerance
    X = []
    Theta = [[1.5, 1], [5, 1], [-1.5, 1]]
    for theta in Theta:
        X += simulated_gaussian_cluster(N, theta[0], theta[1], m, n)
    nodes = random_split_over_nodes(X, s)
    [muk, sigk] = dgmm_ms_em_remote_initialize_random_gaussian(int((N*k)/s), k)
    ak = [1 for i in range(k)]
    b = True
    i = 0
    D = []
    for node in nodes:
        D += node
    frames = []
    while b:
        print("Iteration %d" % i)
        lM = [[] for i in range(k)]
        wkl = []
        pkl = []
        Nkl = []
        Mkl = []
        for node in nodes:
            pki = dgmm_ms_em_local_gaussian(muk, sigk,node, k)
            wki = dgmm_ms_em_update_local_weights(pki, ak, k)
            Nki = dgmm_ms_em_compute_local_Nk(wki, k)
            Mki = dgmm_ms_em_compute_local_Mk(wki, k)
            pkl += pki
            wkl += wki
            Nkl += Nki
            Mkl += Mki
        Nk = dgmm_ms_em_compute_remote_Nk(Nkl, k)
        Mk = dgmm_ms_em_compute_remote_Mk(Mkl, k)
        for node in nodes:
            ak = dgmm_ms_em_remote_update_alphak(Nk, k)
        muk = dgmm_ms_em_compute_remote_muK(Nk, Mk, k)
        sigl = []
        for i in range(len(nodes)):
            node = nodes[i]
            sigi = dgmm_ms_em_compute_local_sigmaK(node, muk, k, wkl[i])
            sigl += sigi
        sigk = dgmm_ms_em_compute_remote_sigmaK(sigi, Nk, k)
        b = False
        i += 1
    print("Converged")
