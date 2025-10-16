"""
SING Gaze Project, Wieki, University of Vienna
Script author: Pierre Labendzki
June 2024

This script plots the permutation analyses of look-related changes
in audio features. 

"""

# import os
import matplotlib.pyplot as plt
# import researchpy as rp
# import pandas as pd
import numpy as np
# import math
# from random import seed
# from random import randint

# from random import seed
from random import randint

import scipy.stats
# from scipy import stats
# from csv import writer
# import random as rd
# from scipy.stats import levene


def cluster_test(ERCs_ORG,ERCs_SUR):
    nb_frames = len(ERCs_ORG[0])

    print('ERCs_ORG', np.shape(ERCs_ORG))
    print('ERCs_SUR', np.shape(ERCs_SUR))

    nb_look = np.shape(ERCs_ORG)[0]
    p_val_continous = np.zeros(nb_frames)
    t_val_continous = np.zeros(nb_frames)
    mean_ORG = np.nanmean(ERCs_ORG,axis = 0)
    mean_SUR = np.nanmean(ERCs_SUR,axis = 0)
    # Distance_to_ORG = np.zeros(nb_frames)
    # Distance_to_SUR = np.zeros(nb_frames)
    for frame in range(nb_frames):
        # print("\n frame = ", frame)
        # print("look = ", look)
        Distance_to_ORG = (ERCs_ORG[:,frame])
        Distance_to_SUR = (ERCs_SUR[:,frame])
        # print("np.shape(Distance_to_ORG)", np.shape(Distance_to_ORG))
        # print("np.shape(Distance_to_SUR)", np.shape(Distance_to_SUR))
            
        t,p = scipy.stats.ttest_ind(Distance_to_ORG, Distance_to_SUR, nan_policy='omit')
        # if (p < 0.05):
        #     print(p)
        #     plt.hist(Distance_to_ORG,alpha = 0.5)
        #     plt.hist(Distance_to_SUR,alpha = 0.5)
        #     plt.title('frame = ' + str(frame))
        #     plt.show()

        p_val_continous[frame] = p
        t_val_continous[frame] = t
    return t_val_continous,p_val_continous    



def cluster(ORG_CORR,SUR_CORR,n_perm):
    n_ORG = np.shape(ORG_CORR)[0]
    n_SUR = np.shape(SUR_CORR)[0]
    n_COL = n_ORG + n_SUR

    COL_CORR = np.vstack((ORG_CORR,SUR_CORR))
    COL_CORR_GROUND =COL_CORR
    random_p_distributions = np.zeros(np.shape(ORG_CORR)[1])
    
    for perm in range(n_perm):
        print("perm = ", perm)
        rdn_ORG = np.zeros(np.shape(ORG_CORR)[1])
        rdn_SUR = np.zeros(np.shape(SUR_CORR)[1])
        COL_CORR_temp = COL_CORR_GROUND
        for k in range(n_ORG):
            value = randint(0, n_COL-k-1)
            # print(value)
            COL_CORR_temp = np.delete(COL_CORR_temp, value, axis=0)
            rdn_SUR = COL_CORR_temp
            rdn_ORG = np.vstack((rdn_ORG,COL_CORR[value,:]))

            # print("rdn_ORG = ", np.shape(rdn_ORG))
            # print("rdn_SUR = ", np.shape(rdn_SUR))

        rdn_ORG = np.delete(rdn_ORG, (0), axis=0) 
        rdn_ORG = np.delete(rdn_ORG, (0), axis=0)  
            
        t_val_continous,p_val_continous = cluster_test(rdn_ORG,rdn_SUR)
        random_p_distributions = np.vstack((random_p_distributions,p_val_continous))
    random_p_distributions = np.delete(random_p_distributions, (0), axis=0)
    return random_p_distributions

def compare_real_to_cluster(real_p,RDN_CORR,threshold):
    # check if the real observed ORG p value is in the first 5th percentile of the permuted p_values
    N = len(real_p)
    is_significant = np.nan * np.zeros(N) 
    for k in range(N):
        if (real_p[k] < np.percentile(RDN_CORR[:,k], threshold)): ### threshold needs to be 5, NOT 0.05 !!!
            is_significant[k] = 1
    return is_significant




window_size = 5
padd = 0
frame_rate = 100 # frames a second
window_size_samples = window_size * frame_rate

big_path = 'W:/hoehl/projects/sing/Acoustic_analysis_SRE/specflux_python/seconds/new_npy_files/BIG/'

#load files SF
BIG_ERCs_SF_LUL = np.load(big_path+'BIG_ERCs_SF_LUL.npy')
BIG_ERCs_SF_SUR_LUL = np.load(big_path+'BIG_ERCs_SF_SUR_LUL.npy')
BIG_ERCs_SF_PLA = np.load(big_path+'BIG_ERCs_SF_PLA.npy')
BIG_ERCs_SF_SUR_PLA = np.load(big_path+'BIG_ERCs_SF_SUR_PLA.npy')
BIG_ERCs_SF_TOTAL = np.load(big_path+'BIG_ERCs_SF_TOTAL.npy')
BIG_ERCs_SF_SUR_TOTAL = np.load(big_path+'BIG_ERCs_SF_SUR_TOTAL.npy')

BIG_ERCs_SF_PLA = np.delete(BIG_ERCs_SF_PLA, (0), axis=0)
BIG_ERCs_SF_SUR_PLA = np.delete(BIG_ERCs_SF_SUR_PLA, (0), axis=0)
BIG_ERCs_SF_LUL = np.delete(BIG_ERCs_SF_LUL, (0), axis=0)
BIG_ERCs_SF_SUR_LUL= np.delete(BIG_ERCs_SF_SUR_LUL, (0), axis=0)
BIG_ERCs_SF_TOTAL = np.delete(BIG_ERCs_SF_TOTAL, (0), axis=0)
BIG_ERCs_SF_SUR_TOTAL = np.delete(BIG_ERCs_SF_SUR_TOTAL, (0), axis=0)


#ENVELOPE
#load files ENV
BIG_ERCs_ENV_LUL = np.load(big_path+'BIG_ERCs_env_LUL.npy')
BIG_ERCs_ENV_SUR_LUL = np.load(big_path+'BIG_ERCs_env_SUR_LUL.npy')
BIG_ERCs_ENV_PLA = np.load(big_path+'BIG_ERCs_env_PLA.npy')
BIG_ERCs_ENV_SUR_PLA = np.load(big_path+'BIG_ERCs_env_SUR_PLA.npy')
BIG_ERCs_ENV_TOTAL = np.load(big_path+'BIG_ERCs_env_TOTAL.npy')
BIG_ERCs_ENV_SUR_TOTAL = np.load(big_path+'BIG_ERCs_env_SUR_TOTAL.npy')

BIG_ERCs_ENV_PLA = np.delete(BIG_ERCs_ENV_PLA, (0), axis=0)
BIG_ERCs_ENV_SUR_PLA = np.delete(BIG_ERCs_ENV_SUR_PLA, (0), axis=0)
BIG_ERCs_ENV_LUL = np.delete(BIG_ERCs_ENV_LUL, (0), axis=0)
BIG_ERCs_ENV_SUR_LUL = np.delete(BIG_ERCs_ENV_SUR_LUL, (0), axis=0)
BIG_ERCs_ENV_TOTAL = np.delete(BIG_ERCs_ENV_TOTAL, (0), axis=0)
BIG_ERCs_ENV_SUR_TOTAL = np.delete(BIG_ERCs_ENV_SUR_TOTAL, (0), axis=0)

#PITCH
#load files F0
BIG_ERCs_F0_LUL = np.load(big_path+'BIG_ERCs_F0_LUL.npy')
BIG_ERCs_F0_SUR_LUL = np.load(big_path+'BIG_ERCs_F0_SUR_LUL.npy')
BIG_ERCs_F0_PLA = np.load(big_path+'BIG_ERCs_F0_PLA.npy')
BIG_ERCs_F0_SUR_PLA = np.load(big_path+'BIG_ERCs_F0_SUR_PLA.npy')
BIG_ERCs_F0_TOTAL = np.load(big_path+'BIG_ERCs_F0_TOTAL.npy')
BIG_ERCs_F0_SUR_TOTAL = np.load(big_path+'BIG_ERCs_F0_SUR_TOTAL.npy')

BIG_ERCs_F0_PLA = np.delete(BIG_ERCs_F0_PLA, (0), axis=0)
BIG_ERCs_F0_SUR_PLA = np.delete(BIG_ERCs_F0_SUR_PLA, (0), axis=0)
BIG_ERCs_F0_LUL = np.delete(BIG_ERCs_F0_LUL, (0), axis=0)
BIG_ERCs_F0_SUR_LUL = np.delete(BIG_ERCs_F0_SUR_LUL, (0), axis=0)
BIG_ERCs_F0_TOTAL = np.delete(BIG_ERCs_F0_TOTAL, (0), axis=0)
BIG_ERCs_F0_SUR_TOTAL = np.delete(BIG_ERCs_F0_SUR_TOTAL, (0), axis=0)



### COMPUTE MEAN AND STD FOR ORG AND SUR
## SF PLA
mean_SF_PLA = np.nanmean(BIG_ERCs_SF_PLA,axis = 0)
sigma_SF_PLA = scipy.stats.sem(BIG_ERCs_SF_PLA,axis = 0)
mean_SF_SUR_PLA = np.nanmean(BIG_ERCs_SF_SUR_PLA,axis = 0)
sigma_SF_SUR_PLA = scipy.stats.sem(BIG_ERCs_SF_SUR_PLA,axis = 0)

## SF LUL
mean_SF_LUL = np.nanmean(BIG_ERCs_SF_LUL,axis = 0)
sigma_SF_LUL = scipy.stats.sem(BIG_ERCs_SF_LUL,axis = 0)
mean_SF_SUR_LUL = np.nanmean(BIG_ERCs_SF_SUR_LUL,axis = 0)
sigma_SF_SUR_LUL = scipy.stats.sem(BIG_ERCs_SF_SUR_LUL,axis = 0)

## SF ALL
mean_SF_TOTAL = np.nanmean(BIG_ERCs_SF_TOTAL,axis = 0)
sigma_SF_TOTAL = scipy.stats.sem(BIG_ERCs_SF_TOTAL,axis = 0)
mean_SF_SUR_TOTAL = np.nanmean(BIG_ERCs_SF_SUR_TOTAL,axis = 0)
sigma_SF_SUR_TOTAL = scipy.stats.sem(BIG_ERCs_SF_SUR_TOTAL,axis = 0)



#### ENVELOPE
## ENV PLA
mean_ENV_PLA = np.nanmean(BIG_ERCs_ENV_PLA,axis = 0)
sigma_ENV_PLA = scipy.stats.sem(BIG_ERCs_ENV_PLA,axis = 0)
mean_ENV_SUR_PLA = np.nanmean(BIG_ERCs_ENV_SUR_PLA,axis = 0)
sigma_ENV_SUR_PLA = scipy.stats.sem(BIG_ERCs_ENV_SUR_PLA,axis = 0)

## ENV LUL
mean_ENV_LUL = np.nanmean(BIG_ERCs_ENV_LUL,axis = 0)
sigma_ENV_LUL = scipy.stats.sem(BIG_ERCs_ENV_LUL,axis = 0)
mean_ENV_SUR_LUL = np.nanmean(BIG_ERCs_ENV_SUR_LUL,axis = 0)
sigma_ENV_SUR_LUL = scipy.stats.sem(BIG_ERCs_ENV_SUR_LUL,axis = 0)

## ENV ALL
mean_ENV_TOTAL = np.nanmean(BIG_ERCs_ENV_TOTAL,axis = 0)
sigma_ENV_TOTAL = scipy.stats.sem(BIG_ERCs_ENV_TOTAL,axis = 0)
mean_ENV_SUR_TOTAL = np.nanmean(BIG_ERCs_ENV_SUR_TOTAL,axis = 0)
sigma_ENV_SUR_TOTAL = scipy.stats.sem(BIG_ERCs_ENV_SUR_TOTAL,axis = 0)


#### F0
## F0 PLA
mean_F0_PLA = np.nanmean(BIG_ERCs_F0_PLA,axis = 0)
sigma_F0_PLA = scipy.stats.sem(BIG_ERCs_F0_PLA,axis = 0)
mean_F0_SUR_PLA = np.nanmean(BIG_ERCs_F0_SUR_PLA,axis = 0)
sigma_F0_SUR_PLA = scipy.stats.sem(BIG_ERCs_F0_SUR_PLA,axis = 0)

## F0 LUL
mean_F0_LUL = np.nanmean(BIG_ERCs_F0_LUL,axis = 0)
sigma_F0_LUL = scipy.stats.sem(BIG_ERCs_F0_LUL,axis = 0)
mean_F0_SUR_LUL = np.nanmean(BIG_ERCs_F0_SUR_LUL,axis = 0)
sigma_F0_SUR_LUL = scipy.stats.sem(BIG_ERCs_F0_SUR_LUL,axis = 0)

## F0 ALL
mean_F0_TOTAL = np.nanmean(BIG_ERCs_F0_TOTAL,axis = 0)
sigma_F0_TOTAL = scipy.stats.sem(BIG_ERCs_F0_TOTAL,axis = 0)
mean_F0_SUR_TOTAL = np.nanmean(BIG_ERCs_F0_SUR_TOTAL,axis = 0)
sigma_F0_SUR_TOTAL = scipy.stats.sem(BIG_ERCs_F0_SUR_TOTAL,axis = 0)










### PLOTTING RESULTS

window_size = 5 # number of seconds to looks before and after the gaze onset
frame_rate = 100 # frames in a second
window_size_samples = window_size * frame_rate
rt = np.arange(-window_size,window_size,1/frame_rate)



path = 'W:/hoehl/projects/sing/Acoustic_analysis_SRE/specflux_python/seconds/new_npy_files/perm/'
savepath = 'W:/hoehl/projects/sing/Acoustic_analysis_SRE/specflux_python/seconds/new_npy_files/pvalues/'
figpath = 'W:/hoehl/projects/sing/Acoustic_analysis_SRE/specflux_python/seconds/gaze_figs_sec/'

# percentile required by the observed pvalue to be below
p_to_test = 5/2


t_val_sf_pla, p_val_sf_pla = cluster_test(BIG_ERCs_SF_PLA,BIG_ERCs_SF_SUR_PLA)    # Compute ORG p value
RDN_CORR_pla = np.load(path+'RDN_p_SF_PLA.npy')    # Load 1000  permuted p values
# np.save(savepath + "pvalue_SF_PLA.npy",p_val_sf_pla)
# np.save(savepath + "tvalue_SF_PLA.npy",t_val_sf_pla)
p_v_sf_pla = compare_real_to_cluster(p_val_sf_pla,RDN_CORR_pla,p_to_test)   # Check when p value is in 5th/3 percentile
# np.save(savepath + "pv_SF_PLA.npy",p_v_sf_pla) 
p_v_sf_pla_times = np.hstack((p_v_sf_pla[:, np.newaxis], t_val_sf_pla[:, np.newaxis], p_val_sf_pla[:, np.newaxis]))
# np.save(savepath + "p_v_sf_times_PLA.npy", p_v_sf_pla_times)


t_val_sf_lul, p_val_sf_lul = cluster_test(BIG_ERCs_SF_LUL,BIG_ERCs_SF_SUR_LUL)
RDN_CORR_lul = np.load(path+'RDN_p_SF_LUL.npy')
# np.save(savepath + "pvalue_SF_LUL.npy",p_val_sf_lul)
# np.save(savepath + "tvalue_SF_LUL.npy",t_val_sf_lul)
p_v_sf_lul = compare_real_to_cluster(p_val_sf_lul,RDN_CORR_lul,p_to_test)
# np.save(savepath + "pv_SF_LUL.npy",p_v_sf_lul)
p_v_sf_lul_times = np.hstack((p_v_sf_lul[:, np.newaxis], t_val_sf_lul[:, np.newaxis], p_val_sf_lul[:, np.newaxis]))
# np.save(savepath + "p_v_sf_times_LUL.npy", p_v_sf_lul_times)


t_val_sf_total, p_val_sf_total = cluster_test(BIG_ERCs_SF_TOTAL,BIG_ERCs_SF_SUR_TOTAL)
RDN_CORR_total = np.load(path+'RDN_p_SF_TOTAL.npy')
# np.save(savepath + "pvalue_SF_TOTAL.npy",p_val_sf_total)
# np.save(savepath + "tvalue_SF_TOTAL.npy",t_val_sf_total)
p_v_sf_total = compare_real_to_cluster(p_val_sf_total,RDN_CORR_total,p_to_test)
# np.save(savepath + "pv_SF_TOTAL.npy",p_v_sf_total)
p_v_sf_total_times = np.hstack((p_v_sf_total[:, np.newaxis], t_val_sf_total[:, np.newaxis], p_val_sf_total[:, np.newaxis]))
# np.save(savepath + "p_v_sf_times_TOTAL.npy", p_v_sf_total_times)





###### PLOTS POSTER
#       SF TOTAL

line_width = 4
line_height = 1.7

plt.figure(figsize=(10,6))
plt.plot(rt, mean_SF_TOTAL, label='Social Gaze', color='#009e74', linewidth = 3)
plt.fill_between(rt, np.subtract(mean_SF_TOTAL, sigma_SF_TOTAL), np.add(mean_SF_TOTAL, sigma_SF_TOTAL), alpha=0.1, color='#009e74')
plt.plot(rt, mean_SF_SUR_TOTAL, label='Surrogate Gaze', color='#56b4e9', linewidth = 3)
plt.fill_between(rt, np.subtract(mean_SF_SUR_TOTAL, sigma_SF_SUR_TOTAL), np.add(mean_SF_SUR_TOTAL, sigma_SF_SUR_TOTAL), alpha=0.1, color='#56b4e9')
plt.axvline(x=0, color='black', linestyle='--')
plt.plot(rt, line_height * np.mean(BIG_ERCs_SF_TOTAL) * p_v_sf_total, linewidth=line_width, color='#d55e00')
plt.xlabel("time relative to infant gaze onset (s)", fontsize = 20)
plt.ylabel("Spectral Flux of Playsongs and Lullabies", fontsize = 16)
plt.xlim(-5,+5)
plt.ylim((130,800))
plt.text(0.05, 0.93, "(A)", transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom')
line_social_gaze = plt.Line2D([0], [0], color='#009e74', lw=2, label='Social Gaze')
line_surrogate_gaze = plt.Line2D([0], [0], color='#56b4e9', lw=2, label='Surrogate Gaze')
line_p_value = plt.Line2D([0], [0], color='#d55e00', lw=2, label='p < Bonferroni-corrected 5th percentile')
plt.legend(handles=[line_social_gaze, line_surrogate_gaze, line_p_value], loc='upper right')

plt.savefig(figpath + 'SF_total_poster.svg')

# plt.show()

#plot pla
plt.figure(figsize=(10,6))
plt.plot(rt, mean_SF_PLA, label='Social Gaze', color='#009e74', linewidth = 3)
plt.fill_between(rt, np.subtract(mean_SF_PLA, sigma_SF_PLA), np.add(mean_SF_PLA, sigma_SF_PLA), alpha=0.1, color='#009e74')
plt.plot(rt, mean_SF_SUR_PLA, label='Surrogate Gaze', color='#56b4e9', linewidth = 3)
plt.fill_between(rt, np.subtract(mean_SF_SUR_PLA, sigma_SF_SUR_PLA), np.add(mean_SF_SUR_PLA, sigma_SF_SUR_PLA), alpha=0.1, color='#56b4e9')
plt.axvline(x=0, color='black', linestyle='--')
plt.plot(rt, line_height * np.mean(BIG_ERCs_SF_PLA) * p_v_sf_pla, linewidth=line_width, color='#d55e00')
plt.xlabel("time relative to infant gaze onset (s)", fontsize = 20)
plt.ylabel("Spectral Flux of Playsongs", fontsize = 16)
plt.xlim(-5,+5)
plt.ylim((130,800))
plt.text(0.05, 0.93, "(B)", transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom')
plt.savefig(figpath + 'SF_pla_poster.svg')


#plot lul
plt.figure(figsize=(10,6))
plt.plot(rt, mean_SF_LUL, label='Social Gaze', color='#009e74', linewidth = 3)
plt.fill_between(rt, np.subtract(mean_SF_LUL, sigma_SF_LUL), np.add(mean_SF_LUL, sigma_SF_LUL), alpha=0.1, color='#009e74')
plt.plot(rt, mean_SF_SUR_LUL, label='Surrogate Gaze', color='#56b4e9', linewidth = 3)
plt.fill_between(rt, np.subtract(mean_SF_SUR_LUL, sigma_SF_SUR_LUL), np.add(mean_SF_SUR_LUL, sigma_SF_SUR_LUL), alpha=0.1, color='#56b4e9')
plt.axvline(x=0, color='black', linestyle='--')
plt.plot(rt, line_height * np.mean(BIG_ERCs_SF_LUL) * p_v_sf_lul, linewidth=line_width, color='#d55e00')
plt.xlabel("time relative to infant gaze onset (s)", fontsize = 20)
plt.ylabel("Spectral Flux of Lullabies", fontsize = 16)
plt.xlim(-5,+5)
plt.ylim((130,800))
plt.text(0.05, 0.93, "(C)", transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom')
plt.savefig(figpath + 'SF_lul_poster.svg')









###### PLOTS PAPER
#       SF TOTAL

# plt.figure(figsize=(12,6))
# plt.subtitle("Spectral Flux")
# plt.subplot(1,3,1)
# plt.plot(rt,mean_SF_TOTAL, label = 'SF TOTAL',color='#00887A')
# plt.fill_between(rt, np.subtract(mean_SF_TOTAL,sigma_SF_TOTAL), np.add(mean_SF_TOTAL, sigma_SF_TOTAL), alpha = 0.1)
# plt.plot(rt,mean_SF_SUR_TOTAL, label = 'SF SUR TOTAL',color='#9E9E9E')
# plt.fill_between(rt, np.subtract(mean_SF_SUR_TOTAL,sigma_SF_SUR_TOTAL), np.add(mean_SF_SUR_TOTAL, sigma_SF_SUR_TOTAL), alpha = 0.1)
# plt.axvline(x = 0, color = 'black', linestyle = '--')
# plt.plot(rt, line_height*np.mean(BIG_ERCs_SF_TOTAL) * p_v_total,linewidth=line_width, color = '#E74926')
# plt.xlabel("time relative to infant gaze onset")
# plt.ylabel("Spectral Flux")
# # plt.ylim((50,310))
# plt.legend()
# plt.text(0.05, 0.05, "(A)", transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom')

# #plot pla
# # plt.figure(figsize=(12,6))
# # plt.subtitle("Spectral Flux")
# plt.subplot(1,2,1)
# plt.plot(rt,mean_SF_PLA, label='SF PLA', color='#00887A')
# plt.fill_between(rt, np.subtract(mean_SF_PLA,sigma_SF_PLA), np.add(mean_SF_PLA, sigma_SF_PLA), alpha=0.1)
# plt.plot(rt,mean_SF_SUR_PLA, label='SF SUR PLA', color='#9E9E9E')
# plt.fill_between(rt, np.subtract(mean_SF_SUR_PLA,sigma_SF_PLA), np.add(mean_SF_SUR_PLA, sigma_SF_PLA), alpha=0.1)
# plt.axvline(x=0, color='black', linestyle='--')
# plt.plot(rt, line_height*np.mean(BIG_ERCs_SF_PLA) * p_v_pla, linewidth=line_width, color='#E74926')
# plt.xlabel("time relative to infant gaze onset")
# plt.ylim((50,310))
# # Adding label "(B)" to bottom left
# plt.text(0.05, 0.05, "(A)", transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom')
# plt.legend(loc='upper right')


# #plot lul
# plt.subplot(1,2,2)
# plt.plot(rt,mean_SF_LUL, label = 'SF LUL',color='#00887A')
# plt.fill_between(rt, np.subtract(mean_SF_LUL,sigma_SF_LUL), np.add(mean_SF_LUL, sigma_SF_LUL), alpha = 0.1)
# plt.plot(rt,mean_SF_SUR_LUL, label = 'SF SUR LUL',color='#9E9E9E')
# plt.fill_between(rt, np.subtract(mean_SF_SUR_LUL,sigma_SF_SUR_LUL), np.add(mean_SF_SUR_LUL, sigma_SF_SUR_LUL), alpha = 0.1)
# plt.axvline(x = 0, color = 'black', linestyle = '--')
# plt.plot(rt, line_height*np.mean(BIG_ERCs_SF_LUL) * p_v_lul,linewidth=line_width, color = '#E74926')
# plt.xlabel("time relative to infant gaze onset")
# plt.ylim((50,310))
# plt.legend()
# plt.text(0.05, 0.05, "(B)", transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom')




line_width = 4
line_height = 1.7

fig, axs = plt.subplots(1, 2, figsize=(20, 6))

# Plot Playsongs
axs[0].plot(rt, mean_SF_PLA, label='Social Gaze', color='#009e74', linewidth=3)
axs[0].fill_between(rt, mean_SF_PLA - sigma_SF_PLA, mean_SF_PLA + sigma_SF_PLA, alpha=0.1, color='#009e74')
axs[0].plot(rt, mean_SF_SUR_PLA, label='Surrogate Gaze', color='#56b4e9', linewidth=3)
axs[0].fill_between(rt, mean_SF_SUR_PLA - sigma_SF_SUR_PLA, mean_SF_SUR_PLA + sigma_SF_SUR_PLA, alpha=0.1, color='#56b4e9')
axs[0].axvline(x=0, color='black', linestyle='--')
axs[0].plot(rt, line_height * np.mean(BIG_ERCs_SF_PLA) * p_v_sf_pla, linewidth=line_width, color='#d55e00')
axs[0].set_xlabel("time relative to infant gaze onset (s)", fontsize=20)
axs[0].set_ylabel("Spectral Flux of Playsongs", fontsize=16)
axs[0].set_xlim(-5, 5)
axs[0].set_ylim(130, 850)
axs[0].text(0.05, 0.93, "(A)", transform=axs[0].transAxes, fontsize=12, verticalalignment='top')

# Plot Lullabies
line1, = axs[1].plot(rt, mean_SF_LUL, label='Social Gaze', color='#009e74', linewidth=3)
axs[1].fill_between(rt, mean_SF_LUL - sigma_SF_LUL, mean_SF_LUL + sigma_SF_LUL, alpha=0.1, color='#009e74')
line2, = axs[1].plot(rt, mean_SF_SUR_LUL, label='Surrogate Gaze', color='#56b4e9', linewidth=3)
axs[1].fill_between(rt, mean_SF_SUR_LUL - sigma_SF_SUR_LUL, mean_SF_SUR_LUL + sigma_SF_SUR_LUL, alpha=0.1, color='#56b4e9')
axs[1].axvline(x=0, color='black', linestyle='--')
line3, = axs[1].plot(rt, line_height * np.mean(BIG_ERCs_SF_LUL) * p_v_sf_lul, label='p<.05/2', linewidth=line_width, color='#d55e00')
axs[1].set_xlabel("time relative to infant gaze onset (s)", fontsize=20)
axs[1].set_ylabel("Spectral Flux of Lullabies", fontsize=16)
axs[1].set_xlim(-5, 5)
axs[1].set_ylim(130, 850)
axs[1].text(0.05, 0.93, "(B)", transform=axs[1].transAxes, fontsize=12, verticalalignment='top')

# Add legend to the right plot
axs[1].legend(handles=[line1, line2, line3], fontsize=12, loc='upper right')

# plt.tight_layout()
# plt.show()
plt.savefig("W:/hoehl/projects/sing/Acoustic_analysis_SRE/specflux_python/seconds/gaze_figs_sec/SF_plalul_samescale.svg")











### ENVELOPE
t_val_pla, p_val_pla = cluster_test(BIG_ERCs_ENV_PLA,BIG_ERCs_ENV_SUR_PLA)
RDN_CORR_pla = np.load(path+'RDN_p_ENV_PLA.npy')
# np.save(savepath + "pvalue_ENV_PLA.npy",p_val_pla)
# np.save(savepath + "tvalue_ENV_PLA.npy",t_val_pla)
p_v_pla = compare_real_to_cluster(p_val_pla,RDN_CORR_pla,p_to_test)
# np.save(savepath + "pv_ENV_PLA.npy",p_v_pla)
p_v_pla_times = np.hstack((p_v_pla[:, np.newaxis], t_val_pla[:, np.newaxis], p_val_pla[:, np.newaxis]))
# np.save(savepath + "p_v_times_PLA.npy", p_v_pla_times)

t_val_lul, p_val_lul = cluster_test(BIG_ERCs_ENV_LUL,BIG_ERCs_ENV_SUR_LUL)
RDN_CORR_lul = np.load(path+'RDN_p_ENV_LUL.npy')
# np.save("pvalue_ENV_LUL.npy",p_val_lul)
# np.save("tvalue_ENV_LUL.npy",t_val_lul)
p_v_lul = compare_real_to_cluster(p_val_lul,RDN_CORR_lul,p_to_test)
# np.save("pv_ENV_LUL.npy",p_v_lul)
p_v_lul_times = np.hstack((p_v_lul[:, np.newaxis], t_val_lul[:, np.newaxis], p_val_lul[:, np.newaxis]))
# np.save(savepath + "p_v_times_LUL.npy", p_v_lul_times)

t_val_total, p_val_total = cluster_test(BIG_ERCs_ENV_TOTAL,BIG_ERCs_ENV_SUR_TOTAL)
RDN_CORR_total = np.load(path+'RDN_p_ENV_TOTAL.npy')
# np.save("pvalue_ENV_TOTAL.npy",p_val_total)
# np.save("tvalue_ENV_TOTAL.npy",t_val_total)
p_v_total = compare_real_to_cluster(p_val_total,RDN_CORR_total,p_to_test)
# np.save("pv_ENV_TOTAL.npy",p_v_total)
p_v_total_times = np.hstack((p_v_total[:, np.newaxis], t_val_total[:, np.newaxis], p_val_total[:, np.newaxis]))
# np.save(savepath + "p_v_times_TOTAL.npy", p_v_total_times)



#plots
line_width = 4
line_height = 1.2

fig, axs = plt.subplots(1, 2, figsize=(20, 6))

# Plot Playsongs
axs[0].plot(rt, mean_ENV_PLA, label='Social Gaze', color='#009e74', linewidth=3)
axs[0].fill_between(rt, mean_ENV_PLA - sigma_ENV_PLA, mean_ENV_PLA + sigma_ENV_PLA, alpha=0.1, color='#009e74')
axs[0].plot(rt, mean_ENV_SUR_PLA, label='Surrogate Gaze', color='#56b4e9', linewidth=3)
axs[0].fill_between(rt, mean_ENV_SUR_PLA - sigma_ENV_SUR_PLA, mean_ENV_SUR_PLA + sigma_ENV_SUR_PLA, alpha=0.1, color='#56b4e9')
axs[0].axvline(x=0, color='black', linestyle='--')
axs[0].plot(rt, line_height * np.mean(BIG_ERCs_ENV_PLA) * p_v_pla, linewidth=line_width, color='#d55e00')
axs[0].set_xlabel("time relative to infant gaze onset (s)", fontsize=20)
axs[0].set_ylabel("Amplitude Envelope of Playsongs", fontsize=16)
axs[0].set_xlim(-5, 5)
axs[0].set_ylim(80, 190)
axs[0].text(0.05, 0.97, "(A)", transform=axs[0].transAxes, fontsize=12, verticalalignment='top')

# Plot Lullabies
line_height = 1.24

axs[1].plot(rt, mean_ENV_LUL, label='Social Gaze', color='#009e74', linewidth=3)
axs[1].fill_between(rt, mean_ENV_LUL - sigma_ENV_LUL, mean_ENV_LUL + sigma_ENV_LUL, alpha=0.1, color='#009e74')
axs[1].plot(rt, mean_ENV_SUR_LUL, label='Surrogate Gaze', color='#56b4e9', linewidth=3)
axs[1].fill_between(rt, mean_ENV_SUR_LUL - sigma_ENV_SUR_LUL, mean_ENV_SUR_LUL + sigma_ENV_SUR_LUL, alpha=0.1, color='#56b4e9')
axs[1].axvline(x=0, color='black', linestyle='--')
axs[1].plot(rt, line_height * np.mean(BIG_ERCs_ENV_LUL) * p_v_lul, label='p < .05/2', linewidth=line_width, color='#d55e00')
axs[1].set_xlabel("time relative to infant gaze onset (s)", fontsize=20)
axs[1].set_ylabel("Amplitude Envelope of Lullabies", fontsize=16)
axs[1].set_xlim(-5, 5)
axs[1].set_ylim(80, 180)
axs[1].text(0.05, 0.97, "(B)", transform=axs[1].transAxes, fontsize=12, verticalalignment='top')

# Add legend to the right plot
axs[1].legend(handles=[line1, line2, line3], fontsize=12, loc='upper right')

# plt.tight_layout()
# plt.show()
plt.savefig("W:/hoehl/projects/sing/Acoustic_analysis_SRE/specflux_python/seconds/gaze_figs_sec/ENV_plalul_samescale.svg")



# line_height = 1.1

# #total
# plt.figure(figsize=(12,6))
# plt.subtitle("Amplitude Envelope")
# plt.subplot(1,3,1)
# plt.plot(rt,mean_ENV_TOTAL, label = 'ENV TOTAL',color='#00887A')
# plt.fill_between(rt, np.subtract(mean_ENV_TOTAL,sigma_ENV_TOTAL), np.add(mean_ENV_TOTAL, sigma_ENV_TOTAL), alpha = 0.1)
# plt.plot(rt,mean_ENV_SUR_TOTAL, label = 'ENV SUR TOTAL',color='#9E9E9E')
# plt.fill_between(rt, np.subtract(mean_ENV_SUR_TOTAL,sigma_ENV_SUR_TOTAL), np.add(mean_ENV_SUR_TOTAL, sigma_ENV_SUR_TOTAL), alpha = 0.1)
# plt.axvline(x = 0, color = 'black', linestyle = '--')
# plt.plot(rt, line_height*np.mean(BIG_ERCs_ENV_TOTAL) * p_v_total,linewidth=line_width, color = '#E74926')
# plt.xlabel("time relative to infant gaze onset")
# plt.ylabel("Envelope")
# plt.ylim((80,170))
# plt.legend(loc='lower right')
# plt.text(0.05, 0.05, "(A)", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')


# #pla
# plt.subplot(1,3,2)
# plt.plot(rt,mean_ENV_PLA, label = 'ENV PLA',color='#00887A')
# plt.fill_between(rt, np.subtract(mean_ENV_PLA,sigma_ENV_PLA), np.add(mean_ENV_PLA, sigma_ENV_PLA), alpha = 0.1)
# plt.plot(rt,mean_ENV_SUR_PLA, label = 'ENV SUR PLA',color='#9E9E9E')
# plt.fill_between(rt, np.subtract(mean_ENV_SUR_PLA,sigma_ENV_SUR_PLA), np.add(mean_ENV_SUR_PLA, sigma_ENV_SUR_PLA), alpha = 0.1)
# plt.axvline(x = 0, color = 'black', linestyle = '--')
# plt.plot(rt, line_height*np.mean(BIG_ERCs_ENV_PLA) * p_v_pla,linewidth=line_width, color = '#E74926')
# plt.xlabel("time relative to infant gaze onset")
# plt.ylim((80,170))
# plt.legend(loc='lower right')
# plt.text(0.05, 0.05, "(B)", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')


# #lul
# plt.subplot(1,3,3)
# plt.plot(rt,mean_ENV_LUL, label = 'ENV LUL',color='#00887A')
# plt.fill_between(rt, np.subtract(mean_ENV_LUL,sigma_ENV_LUL), np.add(mean_ENV_LUL, sigma_ENV_LUL), alpha = 0.1)
# plt.plot(rt,mean_ENV_SUR_LUL, label = 'ENV SUR LUL',color='#9E9E9E')
# plt.fill_between(rt, np.subtract(mean_ENV_SUR_LUL,sigma_ENV_SUR_LUL), np.add(mean_ENV_SUR_LUL, sigma_ENV_SUR_LUL), alpha = 0.1)
# plt.axvline(x = 0, color = 'black', linestyle = '--')
# plt.plot(rt, line_height*np.mean(BIG_ERCs_ENV_LUL) * p_v_lul,linewidth=line_width, color = '#E74926')
# plt.xlabel("time relative to infant gaze onset")
# plt.ylim((80,170))
# plt.legend(loc='lower right')
# plt.text(0.05, 0.05, "(C)", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')


# plt.tight_layout()
# plt.savefig("W:/hoehl/projects/sing/Acoustic_analysis_SRE/specflux_python/gaze_figs/beats_ENV_all_samescale.svg")
# plt.show()







### F0

t_val_pla, p_val_pla = cluster_test(BIG_ERCs_F0_PLA,BIG_ERCs_F0_SUR_PLA)
RDN_CORR_pla = np.load(path+'RDN_p_F0_PLA.npy')
# np.save(savepath + "pvalue_F0_PLA.npy",p_val_pla)
# np.save(savepath + "tvalue_F0_PLA.npy",t_val_pla)
p_v_pla = compare_real_to_cluster(p_val_pla,RDN_CORR_pla,p_to_test)
# np.save(savepath + "pv_F0_PLA.npy",p_v_pla)
p_v_pla_times = np.hstack((p_v_pla[:, np.newaxis], t_val_pla[:, np.newaxis], p_val_pla[:, np.newaxis]))
# np.save(savepath + "p_v_times_PLA.npy", p_v_pla_times)

t_val_lul, p_val_lul = cluster_test(BIG_ERCs_F0_LUL,BIG_ERCs_F0_SUR_LUL)
RDN_CORR_lul = np.load(path+'RDN_p_F0_LUL.npy')
# np.save("pvalue_F0_LUL.npy",p_val_lul)
# np.save("tvalue_F0_LUL.npy",t_val_lul)
p_v_lul = compare_real_to_cluster(p_val_lul,RDN_CORR_lul,p_to_test)
# np.save("pv_F0_LUL.npy",p_v_lul)
p_v_lul_times = np.hstack((p_v_lul[:, np.newaxis], t_val_lul[:, np.newaxis], p_val_lul[:, np.newaxis]))
# np.save(savepath + "p_v_times_LUL.npy", p_v_lul_times)

t_val_total, p_val_total = cluster_test(BIG_ERCs_F0_TOTAL,BIG_ERCs_F0_SUR_TOTAL)
RDN_CORR_total = np.load(path+'RDN_p_F0_TOTAL.npy')
np.save("pvalue_F0_TOTAL.npy",p_val_total)
np.save("tvalue_F0_TOTAL.npy",t_val_total)
p_v_total = compare_real_to_cluster(p_val_total,RDN_CORR_total,p_to_test)
np.save("pv_F0_TOTAL.npy",p_v_total)
p_v_total_times = np.hstack((p_v_total[:, np.newaxis], t_val_total[:, np.newaxis], p_val_total[:, np.newaxis]))
np.save(savepath + "p_v_times_TOTAL.npy", p_v_total_times)


#plots
line_width = 4
line_height = 1.05

fig, axs = plt.subplots(1, 2, figsize=(20, 6))

# Plot Playsongs
axs[0].plot(rt, mean_F0_PLA, label='Social Gaze', color='#009e74', linewidth=3)
axs[0].fill_between(rt, mean_F0_PLA - sigma_F0_PLA, mean_F0_PLA + sigma_F0_PLA, alpha=0.1, color='#009e74')
axs[0].plot(rt, mean_F0_SUR_PLA, label='Surrogate Gaze', color='#56b4e9', linewidth=3)
axs[0].fill_between(rt, mean_F0_SUR_PLA - sigma_F0_SUR_PLA, mean_F0_SUR_PLA + sigma_F0_SUR_PLA, alpha=0.1, color='#56b4e9')
axs[0].axvline(x=0, color='black', linestyle='--')
axs[0].plot(rt, line_height * np.mean(BIG_ERCs_F0_PLA) * p_v_pla, linewidth=line_width, color='#d55e00')
axs[0].set_xlabel("time relative to infant gaze onset (s)", fontsize=20)
axs[0].set_ylabel("Pitch of Playsongs", fontsize=16)
axs[0].set_xlim(-5, 5)
axs[0].set_ylim(230,270)
axs[0].text(0.05, 0.97, "(A)", transform=axs[0].transAxes, fontsize=12, verticalalignment='top')

# Plot Lullabies
line_height = 1.05

axs[1].plot(rt, mean_F0_LUL, label='Social Gaze', color='#009e74', linewidth=3)
axs[1].fill_between(rt, mean_F0_LUL - sigma_F0_LUL, mean_F0_LUL + sigma_F0_LUL, alpha=0.1, color='#009e74')
axs[1].plot(rt, mean_F0_SUR_LUL, label='Surrogate Gaze', color='#56b4e9', linewidth=3)
axs[1].fill_between(rt, mean_F0_SUR_LUL - sigma_F0_SUR_LUL, mean_F0_SUR_LUL + sigma_F0_SUR_LUL, alpha=0.1, color='#56b4e9')
axs[1].axvline(x=0, color='black', linestyle='--')
axs[1].plot(rt, line_height * np.mean(BIG_ERCs_F0_LUL) * p_v_lul, label='p < .05/2', linewidth=line_width, color='#d55e00')
axs[1].set_xlabel("time relative to infant gaze onset (s)", fontsize=20)
axs[1].set_ylabel("Pitch of Lullabies", fontsize=16)
axs[1].set_xlim(-5, 5)
axs[1].set_ylim(230,270)
axs[1].text(0.05, 0.97, "(B)", transform=axs[1].transAxes, fontsize=12, verticalalignment='top')

# Add legend to the right plot
axs[1].legend(handles=[line1, line2, line3], fontsize=12, loc='upper right')

# plt.tight_layout()
# plt.show()
plt.savefig("W:/hoehl/projects/sing/Acoustic_analysis_SRE/specflux_python/seconds/gaze_figs_sec/F0_plalul_samescale.svg")


# line_height = 1.05

# #total
# plt.figure(figsize=(12,6))
# plt.subtitle("Pitch")
# plt.subplot(1,3,1)
# plt.plot(rt,mean_F0_TOTAL, label = 'F0 TOTAL',color='#00887A')
# plt.fill_between(rt, np.subtract(mean_F0_TOTAL,sigma_F0_TOTAL), np.add(mean_F0_TOTAL, sigma_F0_TOTAL), alpha = 0.1)
# plt.plot(rt,mean_F0_SUR_TOTAL, label = 'F0 SUR TOTAL',color='#9E9E9E')
# plt.fill_between(rt, np.subtract(mean_F0_SUR_TOTAL,sigma_F0_SUR_TOTAL), np.add(mean_F0_SUR_TOTAL, sigma_F0_SUR_TOTAL), alpha = 0.1)
# plt.axvline(x = 0, color = 'black', linestyle = '--')
# plt.plot(rt, line_height*np.mean(BIG_ERCs_F0_TOTAL) * p_v_total,linewidth=line_width, color = '#E74926')
# plt.xlabel("time relative to infant gaze onset")
# plt.ylabel("Pitch")
# plt.ylim((225,300))
# plt.legend(loc='upper right')
# plt.text(0.05, 0.95, "(A)", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')

# #pla
# plt.subplot(1,3,2)
# plt.plot(rt,mean_F0_PLA, label = 'F0 PLA',color='#00887A')
# plt.fill_between(rt, np.subtract(mean_F0_PLA,sigma_F0_PLA), np.add(mean_F0_PLA, sigma_F0_PLA), alpha = 0.1)
# plt.plot(rt,mean_F0_SUR_PLA, label = 'F0 SUR PLA',color='#9E9E9E')
# plt.fill_between(rt, np.subtract(mean_F0_SUR_PLA,sigma_F0_SUR_PLA), np.add(mean_F0_SUR_PLA, sigma_F0_SUR_PLA), alpha = 0.1)
# plt.axvline(x = 0, color = 'black', linestyle = '--')
# plt.plot(rt, line_height*np.mean(BIG_ERCs_F0_PLA) * p_v_pla,linewidth=line_width, color = '#E74926')
# plt.xlabel("time relative to infant gaze onset")
# plt.ylim((225,300))
# plt.legend(loc='upper right')
# plt.text(0.05, 0.95, "(B)", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')


# #lul
# plt.subplot(1,3,3)
# plt.plot(rt,mean_F0_LUL, label = 'F0 LUL',color='#00887A')
# plt.fill_between(rt, np.subtract(mean_F0_LUL,sigma_F0_LUL), np.add(mean_F0_LUL, sigma_F0_LUL), alpha = 0.1)
# plt.plot(rt,mean_F0_SUR_LUL, label = 'F0 SUR LUL',color='#9E9E9E')
# plt.fill_between(rt, np.subtract(mean_F0_SUR_LUL,sigma_F0_SUR_LUL), np.add(mean_F0_SUR_LUL, sigma_F0_SUR_LUL), alpha = 0.1)
# plt.axvline(x = 0, color = 'black', linestyle = '--')
# plt.plot(rt, line_height*np.mean(BIG_ERCs_F0_LUL) * p_v_lul,linewidth=line_width, color = '#E74926')
# plt.xlabel("time relative to infant gaze onset")
# plt.ylim((225,300))
# plt.legend(loc='upper right')
# plt.text(0.05, 0.95, "(C)", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')


# plt.tight_layout()
# plt.savefig("W:/hoehl/projects/sing/Acoustic_analysis_SRE/specflux_python/gaze_figs/beats_F0_all_samescale.svg")
# plt.show()


