import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
import pandas as pd
from pathlib import Path
import time
import cv2
from scipy import ndimage
from PIL import Image as im
import seaborn as sns

def low_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def gen_spec(file_name, freq_min, freq_max, wsize, th_perc):
    sfreq, data = wavfile.read(file_name)
    print(f"sfreq: {sfreq}")
    # # Apply low-pass filter
    data = low_pass_filter(data, cutoff=120000, fs=sfreq)
    # Define target sampling frequency
    target_freq = 250000

    # Check if resampling is needed
    if sfreq != target_freq:
        # Calculate the resampling factors
        up_factor = target_freq / sfreq
        
        # Resample the data
        data = mne.filter.resample(data, up=up_factor, down=1.0)
        
        # Update the sampling frequency
        sfreq = target_freq

    time = np.arange(len(data))/sfreq
    dt = wsize/sfreq/2

    tf = np.abs(mne.time_frequency.stft(data, wsize=wsize).squeeze())
    freqs = mne.time_frequency.stftfreq(wsize=wsize, sfreq=sfreq)

    stft_df = pd.DataFrame(data=tf[::-1], index=pd.Series(np.round(freqs[::-1]/1000, 1),
                           name="Frequencies (kHz)"), columns=np.arange(tf.shape[1]) * dt)

    plt_dat = stft_df.loc[(stft_df.index > freq_min) &
                          (stft_df.index < freq_max), :]
    v_max = np.percentile(plt_dat, th_perc)

    return stft_df, v_max, sfreq, plt_dat
