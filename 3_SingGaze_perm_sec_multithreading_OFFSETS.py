# -*- coding: utf-8 -*-
"""
SING Gaze Project, Wieki, University of Vienna
Script authors: Pierre Labendzki, Susanne Reisner
June 2024

This script runs permutation analyses of look-related changes in audio features. 
This analysis is for look OFFSETS.

The results of this study have been published as:
"The reciprocal relationship between maternal infant-directed singing and infant gaze"
in Musicae Scientiae, https://doi.org/10.1177/10298649251385676
"""

import os
# import matplotlib.pyplot as plt
# import researchpy as rp
# import pandas as pd
import numpy as np
# import math
# from random import seed
# from random import randint
import scipy.stats
# from scipy import stats
# from csv import writer
# import random as rd
# from scipy.stats import levene
import multiprocessing

def cluster_test(ERCs_ORG, ERCs_SUR):
    nb_frames = len(ERCs_ORG[0])
    p_val_continous = np.zeros(nb_frames)
    t_val_continous = np.zeros(nb_frames)
    mean_ORG = np.nanmean(ERCs_ORG, axis=0)
    mean_SUR = np.nanmean(ERCs_SUR, axis=0)

    for frame in range(nb_frames):
        Distance_to_ORG = ERCs_ORG[:, frame]
        Distance_to_SUR = ERCs_SUR[:, frame]
        t, p = scipy.stats.ttest_ind(Distance_to_ORG, Distance_to_SUR, nan_policy='omit')
        p_val_continous[frame] = p
        t_val_continous[frame] = t
    return t_val_continous, p_val_continous

def cluster_single_permutation(args):
    ORG_CORR, SUR_CORR, n_ORG, n_SUR, n_COL = args
    COL_CORR = np.vstack((ORG_CORR, SUR_CORR))
    COL_CORR_GROUND = COL_CORR.copy()
    np.random.shuffle(COL_CORR)
    rdn_ORG = COL_CORR[:n_ORG]
    rdn_SUR = COL_CORR[n_ORG:]
    t_val_continous, p_val_continous = cluster_test(rdn_ORG, rdn_SUR)
    return p_val_continous

def cluster(ORG_CORR, SUR_CORR, n_perm, processes):
    n_ORG = ORG_CORR.shape[0]
    n_SUR = SUR_CORR.shape[0]
    n_COL = n_ORG + n_SUR
    args_list = [(ORG_CORR, SUR_CORR, n_ORG, n_SUR, n_COL) for _ in range(n_perm)]
    random_p_distributions = []

    def log_result(result):
        random_p_distributions.append(result)
        if len(random_p_distributions) % 10 == 0 or len(random_p_distributions) == n_perm:
            print(f"Completed {len(random_p_distributions)} out of {n_perm} permutations")

    with multiprocessing.Pool(processes=processes) as pool:
        for args in args_list:
            pool.apply_async(cluster_single_permutation, args=(args,), callback=log_result)
        pool.close()
        pool.join()

    return np.array(random_p_distributions)

def compare_real_to_cluster(real_p, RDN_CORR, threshold):
    N = len(real_p)
    is_significant = np.full(N, np.nan)
    for k in range(N):
        if real_p[k] < np.percentile(RDN_CORR[:, k], threshold):
            is_significant[k] = 1
    return is_significant


'''
First, find out how many cores your PC has -> this determines on how many cores
you can run the multiprocessing
'''
os.cpu_count()



''' SF '''

# #SF TOTAL
# if __name__ == "__main__":
#     window_size = 5
#     padd = 0
#     frame_rate = 100
#     window_size_samples = window_size * frame_rate
#     dir_big = '' # your output_dir_big from script 2_SingGaze_generate_BIG_OFFSETS.py
#     BIG_ERCs_SF_TOTAL_OFF = np.load(dir_big + '/BIG_ERCs_SF_TOTAL_OFF.npy')
#     BIG_ERCs_SF_SUR_TOTAL_OFF = np.load(dir_big + '/BIG_ERCs_SF_SUR_TOTAL_OFF.npy')
#     BIG_ERCs_SF_TOTAL_OFF = np.delete(BIG_ERCs_SF_TOTAL_OFF, 0, axis=0)
#     BIG_ERCs_SF_SUR_TOTAL_OFF = np.delete(BIG_ERCs_SF_SUR_TOTAL_OFF, 0, axis=0)

#     nb_perm = 1000
#     path = '' # where you want to save the permutations
#     processes = 10

#     RDN_p_SF_TOTAL_OFF = cluster(BIG_ERCs_SF_TOTAL_OFF, BIG_ERCs_SF_SUR_TOTAL_OFF, nb_perm, processes)
#     np.save(path + 'RDN_p_SF_TOTAL_OFF.npy', RDN_p_SF_TOTAL_OFF)

#SF PLA
if __name__ == "__main__":
    window_size = 5
    padd = 0
    frame_rate = 100
    window_size_samples = window_size * frame_rate
    dir_big = '' # your output_dir_big from script 2_SingGaze_generate_BIG_OFFSETS.py
    BIG_ERCs_SF_PLA_OFF = np.load(dir_big + 'BIG_ERCs_SF_PLA_OFF.npy')
    BIG_ERCs_SF_SUR_PLA_OFF = np.load(dir_big + 'BIG_ERCs_SF_SUR_PLA_OFF.npy')
    BIG_ERCs_SF_PLA_OFF = np.delete(BIG_ERCs_SF_PLA_OFF, 0, axis=0)
    BIG_ERCs_SF_SUR_PLA_OFF = np.delete(BIG_ERCs_SF_SUR_PLA_OFF, 0, axis=0)

    nb_perm = 1000
    path = '' # where you want to save the permutations
    processes = 10  #determine on how many cores you run the analysis

    RDN_p_SF_PLA_OFF = cluster(BIG_ERCs_SF_PLA_OFF, BIG_ERCs_SF_SUR_PLA_OFF, nb_perm, processes)
    np.save(path + 'RDN_p_SF_PLA_OFF.npy', RDN_p_SF_PLA_OFF)

#SF LUL
if __name__ == "__main__":
    window_size = 5
    padd = 0
    frame_rate = 100
    window_size_samples = window_size * frame_rate
    # dir_big = 'W:/hoehl/projects/sing/Acoustic_analysis_SRE/specflux_python/seconds/new_npy_files/BIG/'
    dir_big = 'W:/hoehl/projects/sing/Acoustic_analysis_SRE/specflux_python/seconds/revision/revision_npy_BIG_offsets/' #revision path
    BIG_ERCs_SF_LUL_OFF = np.load(dir_big + 'BIG_ERCs_SF_LUL_OFF.npy')
    BIG_ERCs_SF_SUR_LUL_OFF = np.load(dir_big + 'BIG_ERCs_SF_SUR_LUL_OFF.npy')
    BIG_ERCs_SF_LUL_OFF = np.delete(BIG_ERCs_SF_LUL_OFF, 0, axis=0)
    BIG_ERCs_SF_SUR_LUL_OFF = np.delete(BIG_ERCs_SF_SUR_LUL_OFF, 0, axis=0)

    nb_perm = 1000
    path = '' # where you want to save the permutations
    processes = 18

    RDN_p_SF_LUL_OFF = cluster(BIG_ERCs_SF_LUL_OFF, BIG_ERCs_SF_SUR_LUL_OFF, nb_perm, processes)
    np.save(path + 'RDN_p_SF_LUL_OFF.npy', RDN_p_SF_LUL_OFF)


''' env '''

# #env TOTAL
# if __name__ == "__main__":
#     window_size = 5
#     padd = 0
#     frame_rate = 100
#     window_size_samples = window_size * frame_rate
#     dir_big = '' # your output_dir_big from script 2_SingGaze_generate_BIG_OFFSETS.py
#     BIG_ERCs_env_TOTAL = np.load(dir_big + '/BIG_ERCs_env_TOTAL.npy')
#     BIG_ERCs_env_SUR_TOTAL = np.load(dir_big + '/BIG_ERCs_env_SUR_TOTAL.npy')
#     BIG_ERCs_env_TOTAL = np.delete(BIG_ERCs_env_TOTAL, 0, axis=0)
#     BIG_ERCs_env_SUR_TOTAL = np.delete(BIG_ERCs_env_SUR_TOTAL, 0, axis=0)

#     nb_perm = 1000
#     path = '' # where you want to save the permutations
#     processes = 12

#     RDN_p_env_TOTAL = cluster(BIG_ERCs_env_TOTAL, BIG_ERCs_env_SUR_TOTAL, nb_perm, processes)
#     np.save(path + 'RDN_p_env_TOTAL.npy', RDN_p_env_TOTAL)

#env PLA
if __name__ == "__main__":
    window_size = 5
    padd = 0
    frame_rate = 100
    window_size_samples = window_size * frame_rate
    # dir_big = 'W:/hoehl/projects/sing/Acoustic_analysis_SRE/specflux_python/seconds/new_npy_files/BIG/'
    dir_big = 'W:/hoehl/projects/sing/Acoustic_analysis_SRE/specflux_python/seconds/revision/revision_npy_BIG_offsets/' #revision path
    BIG_ERCs_env_PLA_OFF = np.load(dir_big + 'BIG_ERCs_env_PLA_OFF.npy')
    BIG_ERCs_env_SUR_PLA_OFF = np.load(dir_big + 'BIG_ERCs_env_SUR_PLA_OFF.npy')
    BIG_ERCs_env_PLA_OFF = np.delete(BIG_ERCs_env_PLA_OFF, 0, axis=0)
    BIG_ERCs_env_SUR_PLA_OFF = np.delete(BIG_ERCs_env_SUR_PLA_OFF, 0, axis=0)

    nb_perm = 1000
    path = '' # where you want to save the permutations
    processes = 18

    RDN_p_env_PLA_OFF = cluster(BIG_ERCs_env_PLA_OFF, BIG_ERCs_env_SUR_PLA_OFF, nb_perm, processes)
    np.save(path + 'RDN_p_env_PLA_OFF.npy', RDN_p_env_PLA_OFF)

#env LUL
if __name__ == "__main__":
    window_size = 5
    padd = 0
    frame_rate = 100
    window_size_samples = window_size * frame_rate
    dir_big = '' # your output_dir_big from script 2_SingGaze_generate_BIG_OFFSETS.py
    BIG_ERCs_env_LUL_OFF = np.load(dir_big + 'BIG_ERCs_env_LUL_OFF.npy')
    BIG_ERCs_env_SUR_LUL_OFF = np.load(dir_big + 'BIG_ERCs_env_SUR_LUL_OFF.npy')
    BIG_ERCs_env_LUL_OFF = np.delete(BIG_ERCs_env_LUL_OFF, 0, axis=0)
    BIG_ERCs_env_SUR_LUL_OFF = np.delete(BIG_ERCs_env_SUR_LUL_OFF, 0, axis=0)

    nb_perm = 1000
    path = '' # where you want to save the permutations
    processes = 18

    RDN_p_env_LUL_OFF = cluster(BIG_ERCs_env_LUL_OFF, BIG_ERCs_env_SUR_LUL_OFF, nb_perm, processes)
    np.save(path + 'RDN_p_env_LUL_OFF.npy', RDN_p_env_LUL_OFF)




''' F0 '''

# #F0 TOTAL
# if __name__ == "__main__":
#     window_size = 5
#     padd = 0
#     frame_rate = 100
#     window_size_samples = window_size * frame_rate
#     dir_big = '' # your output_dir_big from script 2_SingGaze_generate_BIG_OFFSETS.py
#     BIG_ERCs_F0_TOTAL = np.load(dir_big + '/BIG_ERCs_F0_TOTAL.npy')
#     BIG_ERCs_F0_SUR_TOTAL = np.load(dir_big + '/BIG_ERCs_F0_SUR_TOTAL.npy')
#     BIG_ERCs_F0_TOTAL = np.delete(BIG_ERCs_F0_TOTAL, 0, axis=0)
#     BIG_ERCs_F0_SUR_TOTAL = np.delete(BIG_ERCs_F0_SUR_TOTAL, 0, axis=0)

#     nb_perm = 1000
#     path = '' # where you want to save the permutations
#     processes = 12

#     RDN_p_F0_TOTAL = cluster(BIG_ERCs_F0_TOTAL, BIG_ERCs_F0_SUR_TOTAL, nb_perm, processes)
#     np.save(path + 'RDN_p_F0_TOTAL.npy', RDN_p_F0_TOTAL)

#F0 PLA
if __name__ == "__main__":
    window_size = 5
    padd = 0
    frame_rate = 100
    window_size_samples = window_size * frame_rate
    dir_big = '' # your output_dir_big from script 2_SingGaze_generate_BIG_OFFSETS.py
    BIG_ERCs_F0_PLA_OFF = np.load(dir_big + 'BIG_ERCs_F0_PLA_OFF.npy')
    BIG_ERCs_F0_SUR_PLA_OFF = np.load(dir_big + 'BIG_ERCs_F0_SUR_PLA_OFF.npy')
    BIG_ERCs_F0_PLA_OFF = np.delete(BIG_ERCs_F0_PLA_OFF, 0, axis=0)
    BIG_ERCs_F0_SUR_PLA_OFF = np.delete(BIG_ERCs_F0_SUR_PLA_OFF, 0, axis=0)

    nb_perm = 1000
    path = '' # where you want to save the permutations
    processes = 18

    RDN_p_F0_PLA_OFF = cluster(BIG_ERCs_F0_PLA_OFF, BIG_ERCs_F0_SUR_PLA_OFF, nb_perm, processes)
    np.save(path + 'RDN_p_F0_PLA_OFF.npy', RDN_p_F0_PLA_OFF)

#F0 LUL
if __name__ == "__main__":
    window_size = 5
    padd = 0
    frame_rate = 100
    window_size_samples = window_size * frame_rate
    dir_big = '' # your output_dir_big from script 2_SingGaze_generate_BIG_OFFSETS.py
    BIG_ERCs_F0_LUL_OFF = np.load(dir_big + 'BIG_ERCs_F0_LUL_OFF.npy')
    BIG_ERCs_F0_SUR_LUL_OFF = np.load(dir_big + 'BIG_ERCs_F0_SUR_LUL_OFF.npy')
    BIG_ERCs_F0_LUL_OFF = np.delete(BIG_ERCs_F0_LUL_OFF, 0, axis=0)
    BIG_ERCs_F0_SUR_LUL_OFF = np.delete(BIG_ERCs_F0_SUR_LUL_OFF, 0, axis=0)

    nb_perm = 1000
    path = '' # where you want to save the permutations
    processes = 12

    RDN_p_F0_LUL_OFF = cluster(BIG_ERCs_F0_LUL_OFF, BIG_ERCs_F0_SUR_LUL_OFF, nb_perm, processes)
    # np.save(path + 'RDN_p_F0_LUL_OFF.npy', RDN_p_F0_LUL_OFF)
