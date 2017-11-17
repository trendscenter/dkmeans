#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 12:59:51 2017

@author: bbaker
"""
import matplotlib.pyplot as plt
import numpy as np
import imageio


def save_frames_to_gif(framenames, gifname):
    images = []
    for filename in framenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(gifname, images, duration=1.5)


def simulated_gaussian_cluster(N, mu, sigsqr, m, n):
    x = []
    for i in range(N):
        x += [sigsqr * np.random.randn(m, n).flatten() + mu]
    return x


def random_split_over_nodes(X, s):
    f = int(np.floor(len(X) / s))
    indices = np.random.choice(len(X), size=[s, f])
    D = []
    for index in indices:
        d = []
        for i in index:
            # print(i)
            d += [X[i]]
        D += [d]
    r = int(np.ceil(len(X) / s)) - f
    for index in range(r-1, -1, -1):
        D[-1] += [X[-index]]
    findices = [item for sublist in indices for item in sublist]
    return D, findices


def show_centroids(w, thresh=[-0.4, 0.4], cmap='jet'):
    fig, axs = plt.subplots(1, len(w))
    for i, o in enumerate(w):
        ax = axs[i]
        im = ax.imshow(w[i], interpolation='none', vmin=min(thresh),
                       vmax=max(thresh), cmap=cmap)
    plt.colorbar(im)
    plt.show()
    return fig


def plot_clustering(D, C, w, k, s, N, xlab='x', ylab='y', title='', legend=[]):
    markers = ["v", "^", "<", ">", "1", "2", "3", "4", "8",
               "s", "p", "*", "h", "D"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(s):
        start = i * N + 1 * (i > 0)
        end = i*N + N
        x = [d[:][0] for d in D[start:end]]
        y = [d[:][1] for d in D[start:end]]
        scatter = ax.scatter(x, y, c=np.array(C[start:end]),
                             cmap='spring', s=50, marker=markers[i])
    x = [wi[0][0] for wi in w]
    y = [wi[0][1] for wi in w]
    scatter = ax.scatter(x, y, c=np.array(range(k)),
                         cmap='spring', s=100, marker="o")
    plt.colorbar(scatter)
    plt.legend(legend)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.show()
    return fig


def plot_clustering_node_subplots(Df, Cf, W, k, s, xlab='x',
                                  ylab='y', title=''):
    markers = ["v", "^", "<", ">", "1", "2", "3", "4", "8",
               "s", "p", "*", "h", "D"]
    fig, axs = plt.subplots(1, s)
    for i in range(s):
        w = W[i]
        D = Df[i]
        C = Cf[i]
        ax = axs[i]
        ax.set_xlabel(xlab[i % len(xlab)])
        ax.set_ylabel(ylab[i % len(ylab)])
        ax.set_title(title[i % len(title)])
        x = [d[0] for d in D]
        y = [d[1] for d in D]
        scatter = ax.scatter(x, y, c=np.array(C),
                             cmap='spring', s=50, marker=markers[i])
        x = [wi[0] for wi in w]
        y = [wi[1] for wi in w]
        scatter = ax.scatter(x, y, c=np.array(range(k)),
                             cmap='spring', s=100, marker="o")
    plt.colorbar(scatter)
    plt.show()
    return fig
