"""
SING Gaze Project, Wieki, University of Vienna
Script author: Pierre Labendzki
June 2024

This script creates files of audio features 5 seconds around the onset of
infant social gaze, for both original and surrogate data, at a resolution of 50Hz.

Looks are only included if the audio data 5 seconds before or after the onset of
infant social gaze are has <1 s silence.

Surrogate data are random timepoints in the same audio IF there is no onset of
infant social gaze within 5 seconds AND if there is not >1 s silence in the audio.

The results of this study have been published as:
"The reciprocal relationship between maternal infant-directed singing and infant gaze"
in Musicae Scientiae, https://doi.org/10.1177/10298649251385676
"""



import os
# import librosa
#import audiotools
# import subprocess
# import matplotlib.pyplot as plt
# import researchpy as rp
import pandas as pd
import numpy as np
# import math

# from librosa.effects import time_stretch 

# from scipy import signal
# import scipy.stats
# from scipy import stats
from csv import writer
import random as rd
# from scipy.stats import levene
# from scipy.signal import find_peaks
# from scipy.io import wavfile as io
# from scipy.signal import hilbert, chirp
# from scipy.signal import butter,filtfilt
# from scipy.fftpack import fft
# from scipy.fftpack import rfft

# from scipy.stats import entropy
# #relative entropy = KLdiv
# from scipy.special import rel_entr

from filtering import *

# from random import seed
# from random import randint

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def interpolate_nan(array_like):
    # interpolate missing values
    array = array_like.copy()
    array = np.nan_to_num(array, copy=True, nan=np.nan, posinf=np.nan, neginf=np.nan)
    isnan_array = ~np.isnan(array)
    xp = isnan_array.ravel().nonzero()[0]
    fp = array[~np.isnan(array)]
    x = np.isnan(array).ravel().nonzero()[0]
    array[np.isnan(array)] = np.interp(x, xp, fp)
    return array

def resize_proportional(arr, n):
    # resize array "arr" to be of size n
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(arr)), arr)


def get_look_validity(ERC): 
    # check if ERC contains less than a second of NaN
    size_look = len(ERC)
    threshold = 100  # 100 x 10ms (sample rate)
    copy_ERC = ERC
    if (np.count_nonzero(np.isnan(ERC))<=threshold):
         return True
    else:
        return False

def gen_ERC(time_serie, look_onsets, window_size):
    #Generate "ERC" (i.e) timeseries sub-arrays for size window_size, locked on the looks contained in look_onsets

    frame_rate = 100 # frames a second
    window_size_samples = window_size * frame_rate
    # initialise output to be 2*100*duration. in our case 600 samples = 2*100*3s
    ERCs = np.zeros(2*window_size_samples)
    for look in look_onsets:

        #get the local tempo for that song ans paricipant at that time - not properly working in python!
            # we got song tempi with the MIRtoolbox. this does not allow for sliding window analysis,
            # which we would need for the SF analysis        # tempo = tempi[int(100*look)]
        # print('tempo = ' ,tempo)
        # ref_tempo = 120  
        # # stretch the time dimension so that every ERC contains exactly 3beats before and after look onset
        # strech_ratio = ref_tempo / tempo
        #Each ERC is the timeserie value truncated around the look onset + or - the tempo-adjusted window_size
        ERC = time_serie[int(100*look) - int(window_size_samples) : int(100*look) + int(window_size_samples)]
        
        ## to plot each ERC
        # plt.plot(ERC)
        # plt.axvline(x = int(len(ERC)/2), color = 'black', linestyle = '--')
        # plt.legend()
        # plt.show()
        ##

        # if ERC is not empty
        # commented 03/03/2025
        # if (abs(np.sum(ERC)) > 0.0001):
        #     # resize ERC to be 600samples (time streatching occurs here)
        #     #if look is valid
        #     if ((len(ERC) == window_size_samples * 2) and (get_look_validity(ERC))):
        #         #stack the present ERC with the previosu ones
        #         ERCs = np.vstack((ERCs, ERC))
        #     else:
        #         print("Size or validity is NOT correct")
            # resize ERC to be 600samples (time streatching occurs here)
            #if look is valid
            
            
        if ((len(ERC) == window_size_samples * 2) and (get_look_validity(ERC))):
            #stack the present ERC with the previosu ones
            ERCs = np.vstack((ERCs, ERC))
        # else:
        #     print("Size or validity is NOT correct")

    if (len(ERCs.shape)>1):
        #remove first initialised (empty) ERCs line
        ERCs = np.delete(ERCs, (0), axis=0)    
    return ERCs


def toBinary(vad,threshold): #transform to binary (if value is below threshold then set to one)
    vad[vad > threshold] = np.nan
    vad[vad < threshold] = 1
    return vad

def naning_silent(a,t):
    # set to NaN value in a below a threshold t
    a[a < t] = np.nan
    return a


def toBinaryBool(vad,threshold):
    vad[vad > threshold] = np.nan
    vad[vad < threshold] = 1
    return vad  

def get_look_list(look_file): #get the look list in the playsong and lullaby condition
    dfl = pd.read_excel(look_file)
    dfl_pla = dfl[dfl['Condition'] == 'pla']
    dfl_lul = dfl[dfl['Condition'] == 'lul']
    looks_pla = (dfl_pla['Onset']).to_numpy()
    looks_lul = (dfl_lul['Onset']).to_numpy()
    return looks_pla, looks_lul

def get_onset_songs(dfse,ppt_id): #get the start and end of the playsong and the lullaby
    se = dfse[dfse['ID'] == int(ppt_id)]
    print(se)
    start_pla = se['Start_PLA']
    end_pla = se['End_PLA']
    start_lul =se['Start_LUL']
    end_lul = se['End_LUL']
    return start_pla,end_pla, start_lul, end_lul

def clean_SUR_looks(looks,looks_sur): # remove SURrogate looks that are within 3s of an actual original look
    n_looks_sur = len(looks_sur)
    cleaned_looks_sur = []
    for i in range(n_looks_sur):
        if (min([abs(looks_sur[i]-look) for look in looks]) > 5):
            cleaned_looks_sur = np.append(cleaned_looks_sur,looks_sur[i])
    return cleaned_looks_sur

def make_sure_enough_looks(looks): #make sur that the look list is at least threshold size. 
    # If not then copy the look list until achived, and then randomly select the required number of looks to return.
    nb_sur_look = len(looks)
    threshold = 1600
    print("nb sur looks", nb_sur_look)
    if (nb_sur_look<threshold):
        new_size = len(looks)
        while(new_size<threshold):
            looks = np.append(looks, looks)
            new_size = len(looks)
            print('looks mreplicated')
        return rd.sample(list(looks), threshold)
    else:
        return looks

def make_sure_enough_ERCs(ERCs): #make sur that the ERC list is at least threshold size. 
    # If not then copy the ERC array until achieved, and then randomly select the required number of ERCs to return.
    nb_sur_ERCs = np.shape(ERCs)[0]
    threshold = 1600
    print("nb sur ERCs", nb_sur_ERCs)
    if (nb_sur_ERCs<threshold):
        new_size = np.shape(ERCs)[0]
        while(new_size<threshold):
            ERCs = np.vstack((ERCs, ERCs))
            new_size = np.shape(ERCs)[0]
        return rd.sample(list(ERCs), threshold)
    else:
        return ERCs[:threshold][:]        

    
path_look_file = '' # folder with .xlsx files of looks. files are named "ID_mum.xlsx" and contain the columns ID, Condition, Onset, Offset, and Duration.
# our data can be found here: https://osf.io/dzvhb/overview Folder: Looks_to_mum
look_file_list = os.listdir(path_look_file)
print(look_file_list)
N = len(look_file_list)

df_start_end = pd.read_csv('') #csv file with audiolengths. contains columns ID, Study, Start_LUL, Start_PLA, End_LUL, End_PLA
# for our data, see: https://osf.io/dzvhb/files Logs -> audiolengths.csv


ppt_id_list = ['25','32','34','35','39','42','45','46','49','50','51','52',
               '53','54','56','57','58','60','61','62','63','64','65','66',
               '68','70','71','72','73','76','77','78','80','81','82','83',
               '84','85','86','87','88','89','90','91','92','93','94','97',
               '98','99','100','101','102','103','104','105','106','107','108',
               '109','110','111','112','113','114','115','116','117','118',
               '119','123','124','125','126']

window_size = 5
frame_rate = 100 # frames a second
window_size_samples = window_size * frame_rate

## SF initialised
BIG_ERCs_SF_PLA = np.zeros(2*window_size_samples)
BIG_ERCs_SF_SUR_PLA = np.zeros(2*window_size_samples)
BIG_ERCs_SF_LUL = np.zeros(2*window_size_samples)
BIG_ERCs_SF_SUR_LUL = np.zeros(2*window_size_samples)
# BIG_ERCs_SF_TOTAL = np.zeros(2*window_size_samples)
# BIG_ERCs_SF_SUR_TOTAL = np.zeros(2*window_size_samples)

## envelope initialised
BIG_ERCs_env_PLA = np.zeros(2*window_size_samples)
BIG_ERCs_env_SUR_PLA = np.zeros(2*window_size_samples)
BIG_ERCs_env_LUL = np.zeros(2*window_size_samples)
BIG_ERCs_env_SUR_LUL = np.zeros(2*window_size_samples)
# BIG_ERCs_env_TOTAL = np.zeros(2*window_size_samples)
# BIG_ERCs_env_SUR_TOTAL = np.zeros(2*window_size_samples)

## F0 initialised
BIG_ERCs_F0_PLA = np.zeros(2*window_size_samples)
BIG_ERCs_F0_SUR_PLA = np.zeros(2*window_size_samples)
BIG_ERCs_F0_LUL = np.zeros(2*window_size_samples)
BIG_ERCs_F0_SUR_LUL = np.zeros(2*window_size_samples)
# BIG_ERCs_F0_TOTAL = np.zeros(2*window_size_samples)
# BIG_ERCs_F0_SUR_TOTAL = np.zeros(2*window_size_samples)


#output directory for csv files
output_dir = '' # individual files for each ID
output_dir_big = '' # combined for all IDs

size_total_looks_pla = 0
size_total_looks_lul = 0

# load from the paths from script 1_SingGaze_get_audio_features_beats_commented
# for our data, see: https://osf.io/dzvhb/files audiofeatures -> subfolders SF, enf, pitch

for ppt_id in ppt_id_list:

    print('\nppt_id = ' , ppt_id)
    # Loading each audio features
    SF = np.load(savepath_SF +
        ppt_id + '_SF.npy')

    env = np.load(savepath_env+
        ppt_id + '_env.npy')

    F0 = np.load(savepath_SF_pitch+
        ppt_id + '_pitch.npy')


    F0 = resize_proportional(F0,len(SF))
    env = resize_proportional(env,len(SF))

    SF = naning_silent(SF,0.00000000001) 
    F0 = interpolate_nan(F0)
    env = naning_silent(env, 0.00001)


    look_file =  ppt_id + '_mum.xlsx'
    looks_pla, looks_lul = get_look_list(path_look_file + look_file)
    size_total_looks_pla = size_total_looks_pla + len(looks_pla)
    print("len(looks_pla) = ", len(looks_pla))
    print("size_total_looks_pla = ", size_total_looks_pla)
    size_total_looks_lul = size_total_looks_lul + len(looks_lul)
    print("len(looks_lul) = ", len(looks_lul))
    print("size_total_looks_lul = ", size_total_looks_lul)
    

    start_pla,end_pla, start_lul, end_lul = get_onset_songs(df_start_end,ppt_id)
    start_pla = int(start_pla)
    start_lul = int(start_lul)
    end_pla = int(end_pla)
    end_lul = int(end_lul)

# NaN audio before & after singing conditions
    if(start_pla < start_lul):
        SF[:100*start_pla] = np.nan
        SF[100*end_pla:100*start_lul] = np.nan
        SF[100*end_lul:] = np.nan

        env[:100*start_pla] = np.nan
        env[100*end_pla:100*start_lul] = np.nan
        env[100*end_lul:] = np.nan

        F0[:100*start_pla] = np.nan
        F0[100*end_pla:100*start_lul] = np.nan
        F0[100*end_lul:] = np.nan
    
    if(start_pla > start_lul):
        SF[:100*start_lul] = np.nan
        SF[100*end_lul:100*start_pla] = np.nan
        SF[100*end_pla:] = np.nan 

        env[:100*start_lul] = np.nan
        env[100*end_lul:100*start_pla] = np.nan
        env[100*end_pla:] = np.nan 

        F0[:100*start_lul] = np.nan
        F0[100*end_lul:100*start_pla] = np.nan
        F0[100*end_pla:] = np.nan 


# generate surrogate data looks
    N_random = 1600 # using 1600 to compensate for deletion of segments later
    looks_pla_SUR = np.random.uniform(low = start_pla+1, high= end_pla-1, size=N_random)
    looks_lul_SUR = np.random.uniform(low = start_lul+1, high= end_lul-1, size=N_random)

    if (len(looks_pla) >= 1):
        looks_pla_SUR = clean_SUR_looks(looks_pla,looks_pla_SUR)
    if (len(looks_lul) >= 1):  
        looks_lul_SUR = clean_SUR_looks(looks_lul,looks_lul_SUR)


    looks_pla_SUR = make_sure_enough_looks(looks_pla_SUR)
    looks_lul_SUR = make_sure_enough_looks(looks_lul_SUR)
 
    
    looks_pla_SUR = looks_pla_SUR[:1600]
    looks_lul_SUR = looks_lul_SUR[:1600]


    # t_audio = np.arange(0,int(len(SF))/frame_rate ,1/frame_rate)
    # plt.plot(t_audio, SF, label = 'Spectral flux')
    # [plt.axvline(_x, linewidth=1, color='green') for _x in looks_pla]
    # [plt.axvline(_x, linewidth=0.1, color='red') for _x in looks_pla_SUR]
    
    # [plt.axvline(_x, linewidth=1, color='green') for _x in looks_lul]
    # [plt.axvline(_x, linewidth=0.1, color='red') for _x in looks_lul_SUR]

    # plt.plot(t_audio,env,label = 'envelope')
    # plt.plot(t_audio,F0,label = "pitch")

    # plt.title("look onset Whole songs for ppt = " + ppt_id)
    # plt.xlabel("time")
    # plt.ylabel("SpectralFlux")
    # plt.legend()
    # plt.show()
    
# if there's more than one look -> include audio 
# generation of looks depending on audio feature (SF, env, F0), looks & window size
    
    if (len(looks_pla) >= 1):
        
        #SF
        print("real looks")
        # generate ERC using Spectral FLux, the organic looks and the window size of interest.
        ERC_SF_PLA = gen_ERC(SF,looks_pla,window_size)
        print("Surrogate looks")
        # generate the surrogate ERC using the same function but with the surrogate looks
        ERC_SF_SUR_PLA = gen_ERC(SF,looks_pla_SUR,window_size)

        ERC_SF_SUR_PLA = np.vstack((ERC_SF_SUR_PLA, ERC_SF_SUR_PLA))
        ERC_SF_SUR_PLA = ERC_SF_SUR_PLA[:1000]

        # save every ERC at the participant level
        # np.savetxt(output_dir+'/SF/'+ppt_id+"_SF_org_PLA.csv", ERC_SF_PLA[:1000], delimiter=",")
        # np.savetxt(output_dir+'/SF/'+ppt_id+"_SF_sur_PLA.csv", ERC_SF_SUR_PLA[:1000], delimiter=",")

        # stack all the ERC (across participants together)
        BIG_ERCs_SF_PLA = np.vstack((BIG_ERCs_SF_PLA, ERC_SF_PLA))
        BIG_ERCs_SF_SUR_PLA = np.vstack((BIG_ERCs_SF_SUR_PLA, ERC_SF_SUR_PLA))
    
        # BIG_ERCs_SF_TOTAL = np.vstack((BIG_ERCs_SF_TOTAL, ERC_SF_PLA))
        # BIG_ERCs_SF_SUR_TOTAL = np.vstack((BIG_ERCs_SF_SUR_TOTAL, ERC_SF_SUR_PLA))
    
        ## ENV
        ERC_env_PLA = gen_ERC(env,looks_pla,window_size)
        ERC_env_SUR_PLA = gen_ERC(env,looks_pla_SUR,window_size)

        ERC_env_SUR_PLA = np.vstack((ERC_env_SUR_PLA, ERC_env_SUR_PLA))
        ERC_env_SUR_PLA = ERC_env_SUR_PLA[:1000]
        # np.savetxt(output_dir+'/env/'+ppt_id+"_ENV_org_PLA.csv", ERC_env_PLA[:1000], delimiter=",")
        # np.savetxt(output_dir+'/env/'+ppt_id+"_ENV_sur_PLA.csv", ERC_env_SUR_PLA[:1000], delimiter=",")

        BIG_ERCs_env_PLA = np.vstack((BIG_ERCs_env_PLA, ERC_env_PLA))
        BIG_ERCs_env_SUR_PLA = np.vstack((BIG_ERCs_env_SUR_PLA, ERC_env_SUR_PLA))

        # BIG_ERCs_env_TOTAL = np.vstack((BIG_ERCs_env_TOTAL, ERC_env_PLA))
        # BIG_ERCs_env_SUR_TOTAL = np.vstack((BIG_ERCs_env_SUR_TOTAL, ERC_env_SUR_PLA))

        #F0
        ERC_F0_PLA = gen_ERC(F0,looks_pla,window_size)
        ERC_F0_SUR_PLA = gen_ERC(F0,looks_pla_SUR,window_size)

        ERC_F0_SUR_PLA = np.vstack((ERC_F0_SUR_PLA, ERC_F0_SUR_PLA))
        ERC_F0_SUR_PLA = ERC_F0_SUR_PLA[:1000]

        # np.savetxt(output_dir+'/pitch/'+ppt_id+"_PITCH_org_PLA.csv", ERC_F0_PLA[:1000], delimiter=",")
        # np.savetxt(output_dir+'/pitch/'+ppt_id+"_PITCH_sur_PLA.csv", ERC_F0_SUR_PLA[:1000], delimiter=",")

        BIG_ERCs_F0_PLA = np.vstack((BIG_ERCs_F0_PLA, ERC_F0_PLA))
        BIG_ERCs_F0_SUR_PLA = np.vstack((BIG_ERCs_F0_SUR_PLA, ERC_F0_SUR_PLA))
    
        # BIG_ERCs_F0_TOTAL = np.vstack((BIG_ERCs_F0_TOTAL, ERC_F0_PLA))
        # BIG_ERCs_F0_SUR_TOTAL = np.vstack((BIG_ERCs_F0_SUR_TOTAL, ERC_F0_SUR_PLA))


    if (len(looks_lul) >= 1):
        
        # Doing same as before (PLA), but for the Lullaby
        # SF
        
        print("real looks")
        ERC_SF_LUL = gen_ERC(SF,looks_lul,window_size)
        print("Surrogate looks")
        ERC_SF_SUR_LUL = gen_ERC(SF,looks_lul_SUR,window_size)

        ERC_SF_SUR_LUL = np.vstack((ERC_SF_SUR_LUL, ERC_SF_SUR_LUL))
        ERC_SF_SUR_LUL = ERC_SF_SUR_LUL[:1000]
        # np.savetxt(output_dir+'/SF/'+ppt_id+"_SF_org_LUL.csv", ERC_SF_LUL[:1000], delimiter=",")
        # np.savetxt(output_dir+'/SF/'+ppt_id+"_SF_sur_LUL.csv", ERC_SF_SUR_LUL[:1000], delimiter=",")

        BIG_ERCs_SF_LUL = np.vstack((BIG_ERCs_SF_LUL, ERC_SF_LUL))
        BIG_ERCs_SF_SUR_LUL = np.vstack((BIG_ERCs_SF_SUR_LUL, ERC_SF_SUR_LUL))
    
        # BIG_ERCs_SF_TOTAL = np.vstack((BIG_ERCs_SF_TOTAL, ERC_SF_LUL))
        # BIG_ERCs_SF_SUR_TOTAL = np.vstack((BIG_ERCs_SF_SUR_TOTAL, ERC_SF_SUR_LUL))
    

        ## ENV
        ERC_env_LUL = gen_ERC(env,looks_lul,window_size)
        ERC_env_SUR_LUL = gen_ERC(env,looks_lul_SUR,window_size)

        ERC_env_SUR_LUL = np.vstack((ERC_env_SUR_LUL, ERC_env_SUR_LUL))
        ERC_env_SUR_LUL = ERC_env_SUR_LUL[:1000]
        # np.savetxt(output_dir+'/env/'+ppt_id+"_ENV_org_LUL.csv", ERC_env_LUL[:1000], delimiter=",")
        # np.savetxt(output_dir+'/env/'+ppt_id+"_ENV_sur_LUL.csv", ERC_env_SUR_LUL[:1000], delimiter=",")

        BIG_ERCs_env_LUL = np.vstack((BIG_ERCs_env_LUL, ERC_env_LUL))
        BIG_ERCs_env_SUR_LUL = np.vstack((BIG_ERCs_env_SUR_LUL, ERC_env_SUR_LUL))

        # BIG_ERCs_env_TOTAL = np.vstack((BIG_ERCs_env_TOTAL, ERC_env_LUL))
        # BIG_ERCs_env_SUR_TOTAL = np.vstack((BIG_ERCs_env_SUR_TOTAL, ERC_env_SUR_LUL))

        #F0
        ERC_F0_LUL = gen_ERC(F0,looks_lul,window_size)
        ERC_F0_SUR_LUL = gen_ERC(F0,looks_lul_SUR,window_size)

        ERC_F0_SUR_LUL = np.vstack((ERC_F0_SUR_LUL, ERC_F0_SUR_LUL))
        ERC_F0_SUR_LUL = ERC_F0_SUR_LUL[:1000]
        # np.savetxt(output_dir+'/pitch/'+ppt_id+"_PITCH_org_LUL.csv", ERC_F0_LUL[:1000], delimiter=",")
        # np.savetxt(output_dir+'/pitch/'+ppt_id+"_PITCH_sur_LUL.csv", ERC_F0_SUR_LUL[:1000], delimiter=",")

        BIG_ERCs_F0_LUL = np.vstack((BIG_ERCs_F0_LUL, ERC_F0_LUL))
        BIG_ERCs_F0_SUR_LUL = np.vstack((BIG_ERCs_F0_SUR_LUL, ERC_F0_SUR_LUL))
    
        # BIG_ERCs_F0_TOTAL = np.vstack((BIG_ERCs_F0_TOTAL, ERC_F0_LUL))
        # BIG_ERCs_F0_SUR_TOTAL = np.vstack((BIG_ERCs_F0_SUR_TOTAL, ERC_F0_SUR_LUL))

    print('shape of BIG_ERCs_SF_PLA = ' , np.shape(BIG_ERCs_SF_PLA))
    print('shape of BIG_ERCs_SF_SUR_PLA = ' , np.shape(BIG_ERCs_SF_SUR_PLA))
    print('shape of BIG_ERCs_SF_LUL = ' , np.shape(BIG_ERCs_SF_LUL))
    print('shape of BIG_ERCs_SF_SUR_LUL = ' , np.shape(BIG_ERCs_SF_SUR_LUL))

    print('shape of BIG_ERCs_env_PLA = ' , np.shape(BIG_ERCs_env_PLA))
    print('shape of BIG_ERCs_env_SUR_PLA = ' , np.shape(BIG_ERCs_env_SUR_PLA))
    print('shape of BIG_ERCs_env_LUL = ' , np.shape(BIG_ERCs_env_LUL))
    print('shape of BIG_ERCs_env_SUR_LUL = ' , np.shape(BIG_ERCs_env_SUR_LUL))
    
    print('shape of BIG_ERCs_F0_PLA = ' , np.shape(BIG_ERCs_F0_PLA))
    print('shape of BIG_ERCs_F0_SUR_PLA = ' , np.shape(BIG_ERCs_F0_SUR_PLA))
    print('shape of BIG_ERCs_F0_LUL = ' , np.shape(BIG_ERCs_F0_LUL))
    print('shape of BIG_ERCs_F0_SUR_LUL = ' , np.shape(BIG_ERCs_F0_SUR_LUL))






np.save(output_dir_big+"BIG_ERCs_SF_PLA_ON.npy", BIG_ERCs_SF_PLA[:])
np.save(output_dir_big+"BIG_ERCs_SF_SUR_PLA_ON.npy", BIG_ERCs_SF_SUR_PLA[:])

np.save(output_dir_big+"BIG_ERCs_SF_LUL_ON.npy", BIG_ERCs_SF_LUL[:])
np.save(output_dir_big+"BIG_ERCs_SF_SUR_LUL_ON.npy", BIG_ERCs_SF_SUR_LUL[:])

# np.save(output_dir_big+"BIG_ERCs_SF_TOTAL_ON.npy", BIG_ERCs_SF_TOTAL[:])
# np.save(output_dir_big+"BIG_ERCs_SF_SUR_TOTAL_ON.npy", BIG_ERCs_SF_SUR_TOTAL[:])



np.save(output_dir_big+"BIG_ERCs_env_PLA_ON.npy", BIG_ERCs_env_PLA)
np.save(output_dir_big+"BIG_ERCs_env_SUR_PLA_ON.npy", BIG_ERCs_env_SUR_PLA)

np.save(output_dir_big+"BIG_ERCs_env_LUL_ON.npy", BIG_ERCs_env_LUL)
np.save(output_dir_big+"BIG_ERCs_env_SUR_LUL_ON.npy", BIG_ERCs_env_SUR_LUL)

# np.save(output_dir_big+"BIG_ERCs_env_TOTAL_ON.npy", BIG_ERCs_env_TOTAL)
# np.save(output_dir_big+"BIG_ERCs_env_SUR_TOTAL_ON.npy", BIG_ERCs_env_SUR_TOTAL)



np.save(output_dir_big+"BIG_ERCs_F0_PLA_ON.npy", BIG_ERCs_F0_PLA)
np.save(output_dir_big+"BIG_ERCs_F0_SUR_PLA_ON.npy", BIG_ERCs_F0_SUR_PLA)

np.save(output_dir_big+"BIG_ERCs_F0_LUL_ON.npy", BIG_ERCs_F0_LUL)
np.save(output_dir_big+"BIG_ERCs_F0_SUR_LUL_ON.npy", BIG_ERCs_F0_SUR_LUL)

# np.save(output_dir_big+"BIG_ERCs_F0_TOTAL_ON.npy", BIG_ERCs_F0_TOTAL)
# np.save(output_dir_big+"BIG_ERCs_F0_SUR_TOTAL_ON.npy", BIG_ERCs_F0_SUR_TOTAL)






