# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:28:04 2018

@author: Thomas Schatz

Compute distances, scores and results given task and features.

Usage: 
    python run_abx.py feat_file task_file res_folder res_id distance normalized
    
    where 'distance' is 'kl' or 'cos' and 'normalized' is a boolean
"""


# change to distance to force choice of normalization is crazy and crazily done
# this should not affect the ABXpy code in the slightest...
# the name of the optional argument is even forced!!!

import ABXpy.distances.distances as dis
import ABXpy.score as sco
import ABXpy.analyze as ana
#import ABXpy.distances.metrics.dtw as dtw
#import ABXpy.distances.metrics.kullback_leibler as kl
#import ABXpy.distances.metrics.cosine as cos
import scipy.spatial.distance as scipy_dis
import numpy as np
import dtw
import cosine


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
    metric1 = lambda x, y: scipy_dis.cdist(x, y, 'euclidean')  # cosine on 1D data does not make sense
    if cosine_type == 'angular':
        metric2 = cosine.matched_angular
    elif cosine_type == 'cosine':
        metric2 = cosine.matched_cosine
    else:
        raise ValueError('Unsupported cosine_type value: {}'.format(cosine_type))
    return alignment_then_diss(x[:, :1], x[:, 1:], y[:, :1], y[:, 1:],
                               metric1, metric2, normalized=normalized)


def run_ABX(feat_file, task_file, dis_file, score_file, result_file, distance,
            normalized):
    """
    Run distances, scores and results ABXpy steps based on
    provided features and task files.
    Results are saved in:
        $res_folder/distances/'$res_id'.distances
        $res_folder/scores/'$res_id'.scores
        $res_folder/results/'$res_id'.txt
    """
    dis.compute_distances(feat_file, '/features/', task_file, dis_file,
                          distance, normalized=normalized, n_cpu=1)
    sco.score(task_file, dis_file, score_file)
    ana.analyze(task_file, score_file, result_file)


if __name__=='__main__':
    import argparse
    import os.path as path
    parser = argparse.ArgumentParser()
    parser.add_argument('feat_file', help='h5features file')
    parser.add_argument('task_file', help='ABXpy task file')
    parser.add_argument('res_folder', help=('Result folder (must contain'
                                            'distances, scores and results'
                                            'subfolders)'))
    parser.add_argument('res_id', help=('identifier for the results'
                                        '(model + task)'))
    parser.add_argument('distance', help='dtw-cos, dtw-ang, dtw-logE+cos, dtw-logE+ang')
    parser.add_argument('normalized')
    args = parser.parse_args()
    assert path.exists(args.feat_file), ("No such file "
                                         "{}".format(args.feat_file))
    assert path.exists(args.task_file), ("No such file "
                                         "{}".format(args.task_file))
    dis_file = path.join(args.res_folder, 'distances',
                         args.res_id + '.distances')    
    score_file = path.join(args.res_folder, 'scores',
                           args.res_id + '.scores')
    result_file = path.join(args.res_folder, 'results',
                            args.res_id + '.txt')
    assert not(path.exists(dis_file)), \
        "{} already exists".format(dis_file)
    assert not(path.exists(score_file)), \
        "{} already exists".format(score_file)
    assert not(path.exists(result_file)), \
        "{} already exists".format(result_file)
    assert args.distance in ['dtw-cos', 'dtw-ang',
                             'dtw-logE+cos', 'dtw-logE+ang'], \
        "Distance function {} not supported".format(args.distance)
    if args.distance == 'dtw-cos':
        frame_dis = cosine.all_cosine
    #    elif args.distance == 'dtw-euc':
    #        frame_dis = lambda x, y: scipy_dis.cdist(x, y, 'euclidean')
    elif args.distance == 'dtw-ang':
        frame_dis = cosine.all_angular
    elif args.distance == 'dtw-logE+cos':
        cosine_type = 'cosine'
    elif args.distance == 'dtw-logE+ang':
        cosine_type = 'angular'

    if not('logE' in args.distance):
        distance = lambda x, y, normalized: dtw.dtw(x, y, frame_dis,
                                                    normalized=normalized)
    else: 
        distance = lambda x, y, normalized: dtw_on_logE(x, y, normalized,
                                                        cosine_type=cosine_type)
    if args.normalized == 'True':
        normalized = True
    elif args.normalized == 'False':
        normalized = False
    else:
        raise ValueError('Unsupported normalized value {}'.format(normalized))
    run_ABX(args.feat_file, args.task_file, dis_file, score_file, result_file,
            distance, normalized)

