# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 01:47:42 2014

@author: Thomas Schatz
"""
import numpy as np
import scipy


def _all_cosine_d(x, y, s2d):
    """
    Compute cosine/angular "distances" between all possible pairs of lines in the x and y matrix
    x and y should be 2D numpy arrays with respectively n_x and n_y lines and a common
    number d of columns (feature dimension)
    x, y must be float arrays
    returns a size n_x, n_y matrix
    """
    assert (x.dtype == np.float64 and y.dtype == np.float64) or (
        x.dtype == np.float32 and y.dtype == np.float32)
    x2 = np.sqrt(np.sum(x ** 2, axis=1))
    y2 = np.sqrt(np.sum(y ** 2, axis=1))
    ix = x2 == 0.
    iy = y2 == 0.
    s = np.dot(x, y.T) / (np.outer(x2, y2))
    d = s2d(s)
    d[ix, :] = 1.
    d[:, iy] = 1.
    for i in np.where(ix)[0]:
        d[i, iy] = 0.
    assert np.all(d >= 0)
    return d


def _matched_cosine_d(x, y, s2d):
    """
    Compute cosine/angular "distances" between matched pairs of lines in the x and y matrix
    x and y should be 2D numpy arrays with same number n of lines and same
    number d of columns (feature dimension)
    x, y must be float arrays
    returns a size n vector
    """
    assert (x.dtype == np.float64 and y.dtype == np.float64) or (
        x.dtype == np.float32 and y.dtype == np.float32)
    x2 = np.sqrt(np.sum(x ** 2, axis=1))
    y2 = np.sqrt(np.sum(y ** 2, axis=1))
    ix = x2 == 0.
    iy = y2 == 0.
    # element-wise operations
    s = np.sum(x * y, axis=1) / (x2 * y2)
    d = s2d(s)
    d[ix] = 1.
    d[iy] = 1.
    d[ix & iy] = 0.
    assert np.all(d >= 0)
    return d


def s2d_cosine(s):
    return 1. - s


def s2d_angular(s):
    if s.shape == (1, 1):
        # DPX: to prevent the stupid scipy to collapse the array into scalar
        d = np.array([[np.float64(scipy.arccos(s) / np.pi)]])
    elif s.shape == (1,):
        d = np.array([np.float64(scipy.arccos(s) / np.pi)])
    else:
        # costly in time (half of the time)
        d = np.float64(scipy.arccos(s) / np.pi)
    return d


# return matrix of distances between all possible pairs of lines from x and y
all_cosine = lambda x, y: _all_cosine_d(x, y, s2d_cosine)
all_angular = lambda x, y: _all_cosine_d(x, y, s2d_angular)

# return vector of distances between matched pairs of lines from x and y
matched_cosine =  lambda x, y: _matched_cosine_d(x, y, s2d_cosine)
matched_angular =  lambda x, y: _matched_cosine_d(x, y, s2d_cosine)
