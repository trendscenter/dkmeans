# -*- coding: utf-8 -*-
"""
dkmeans utility functions, not specifically related to clustering
or other activities
"""

import numpy as np


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


