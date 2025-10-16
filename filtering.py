"""
SING Gaze Project, Wieki, University of Vienna
Script author: Pierre Labendzki
June 2024

This script contains various filters.

The results of this study have been published as:
"The reciprocal relationship between maternal infant-directed singing and infant gaze"
in Musicae Scientiae, https://doi.org/10.1177/10298649251385676
"""


import numpy as np
import scipy.signal 
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile as io

import scipy.signal 
from scipy.io import wavfile as io
from scipy.signal import hilbert, chirp
from scipy.signal import butter,filtfilt


def butter_lowpass_filter(data, cutoff, fs, order): ### butterworth lowpass filter
    nyq = fs/2
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order): ### butterworth highpass filter
    nyq = fs/2
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

def filter_signal(signal,sr,LCO,HCO,order): ### butterworth bandpass filter
    signal = butter_lowpass_filter(signal,HCO,sr,order)
    signal = butter_highpass_filter(signal,LCO,sr,order)
    return signal
