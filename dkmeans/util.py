#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Utility functions
"""


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
