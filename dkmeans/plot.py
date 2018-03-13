#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Functions for visualizing centroids or clusterings created with dkmeans
        plot_2d_clustering(nodes, cluster_labels, remote_centroids, k, s, len(X))
"""
import matplotlib.pyplot as plt
import imageio as imo
import numpy as np

MARKERS = ["v", "^", "<", ">", "1", "2", "3", "4", "8",
               "s", "p", "*", "h", "D"]


def save_frames_to_gif(filenames, gifname, duration=1.5):
    """ save a list of separately saved frames to a gif """
    images = [imo.imread(filename) for filename in filenames]
    imo.mimsave(gifname, images, duration=duration)


def show_centroid(centroid, thresh=[-0.4, 0.4], cmap='jet'):
    """Show a 1-d or 2-d centroid using matplotlib imshow"""
    fig, axs = plt.subplots(1, len(centroid))
    for i, o in enumerate(centroid):
        ax = axs[i]
        im = ax.imshow(centroid[i], interpolation='none', vmin=min(thresh),
                       vmax=max(thresh), cmap=cmap)
    plt.colorbar(im)
    plt.show()
    return fig


def plot_2d_clustering(D, C, w, k, s, xlab='x', ylab='y', title='', legend=[]):
    """
        Plot a (2-d) clustering
        TODO: use embeddings to plot higher dimensional clustering
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(s):
        x = [d[:, 0] for d in D[i]]
        y = [d[:, 1] for d in D[i]]
        scatter = ax.scatter(x, y, c=np.array(C[i]).reshape(len(C[i]), 1),
                             cmap='spring', s=50, marker=MARKERS[i])
    x = [wi[0][0] for wi in w]
    y = [wi[0][1] for wi in w]
    scatter = ax.scatter(x, y, c=np.array(range(k)), edgecolor='black',
                         cmap='spring', s=100, marker="o")
    plt.colorbar(scatter)
    plt.legend(legend)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.show()
    return fig
