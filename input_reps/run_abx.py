# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 09:20:00 2019 EST

@author: Thomas Schatz adapted from Gabriel Synaeve's code

The "feature" dimension is along the columns and the "time" dimension along the lines of arrays x and y.
(the user-provided metric function should respect that)
    
The function do not verify its arguments, common problems are:
    shape of one array is n instead of (n,1)
    an array is not of the correct type DTYPE_t 
    the feature dimension of the two array do not match
    the feature and time dimension are exchanged
    the dist_array is not of the correct size or type
"""

import dtw
import cosine
import scipy.spatial.distance as dis




def alignment_then_diss(x1, x2, y1, y2, metric1, metric2, normalized=True):
    """
    Align x1, y1 with DTW based on metric1 and get distance
    using metric2 on x2, y2 along alignment path.
    
        - metric1: returns matrix of distances for all pairs of x1, y1 frames
        - metric2: returns vector of distances for matched pairs of same length
           arrays obtained by aligning x2 and y2
    """
    _, path = dtw.dtw(x1, y1, metric1, return_path=True)
    x = x2[path[:, 0], :]
    y = y2[path[:, 1], :]
    d = metric2(x, y)
    if normalized:
        return np.mean(d)
    else:
        return np.sum(d)
    

def dtw_on_logE(x, y, normalized, cosine_type='angular'):
    """
    x: [n_x, d] array
    y: [n_y, d] array

    This assumes the signal log-energy can be found on the first feature coordinate.
    """
    metric1 = lambda x, y: dis.cdist(x, y, 'euclidean')  # cosine on 1D data does not make sense
    if cosine_type == 'angular':
        metric2 = cosine.matched_angular
    elif cosine_type == 'cosine':
        metric2 = cosine.matched_cosine
    else:
        raise ValueError('Unsupported cosine_type value: {}'.format(cosine_type))
    return alignment_then_diss(x[:, :1], x[:, 1:], y[:, :1], y[:, 1:],
                               metric1, metric2, normalized=normalized)


# dtw_on_E ready to be used in run_abx