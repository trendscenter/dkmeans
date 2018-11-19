#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Utility functions
"""

import numpy as np
import matplotlib.pyplot as plt


def local_maxima(a, x=None, indices=True):
    """
        Finds local maxima in a discrete list 'a' to index an array 'x'
        if 'x' is not specified, 'a' is indexed.
        https://stackoverflow.com/questions/4624970/
                finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
    """
    asm = smooth(a)
    maxima = [asm[i] for i in np.where(np.array(np.r_[1, asm[1:] < asm[:-1]] &
                                       np.r_[asm[:-1] < asm[1:], 1]))[0]]
    matches = [find_nearest(a, maximum) for maximum in maxima]
    indices = [i for i in range(len(a)) if a[i] in matches]
    if indices:
        return matches, indices
    return matches


def smooth(x, window_len=11, window='hanning'):

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning',"
              "'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def split_chunks(l, n):
    """
    Yield successive n-sized chunks from list l.
    https://stackoverflow.com/questions/312443/
                how-do-you-split-a-list-into-evenly-sized-chunks
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def anti_transpose(a):
    """
        Tranpose a matrix on the anti-diagonal
        https://stackoverflow.com/questions/44772451/
            what-is-the-best-way-to-perform-an-anti-transpose-in-python

        This method performs almost twice as fast as when using np transforms
    """
    return a[::-1, ::-1].T
