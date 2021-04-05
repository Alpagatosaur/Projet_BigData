# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 10:10:02 2021

@author: tanch
"""

import librosa
import numpy as np
#import matplotlib.pyplot as plt
import librosa.display
from glob import glob
import csv
import pandas as pd
import matplotlib as plt


path = "Sons/chouette-hulotte-chant-et-cris5.wav"
y,sr = librosa.load(path,sr=None)

fft_y =np.fft.fft(y)
n_fft = len(fft_y)

melfb = librosa.filters.mel(sr, n_fft)

fig, ax = plt.pyplot.subplots()
img = librosa.display.specshow(melfb, x_axis='linear', ax=ax)
ax.set(ylabel='Mel filter', title='Mel filter bank')
#fig.colorbar(img, ax=ax)






mfcc_y = librosa.feature.mfcc(y=y, sr=sr)
fig, ax = plt.pyplot.subplots(nrows=2, sharex=True, sharey=True)
img1 = librosa.display.specshow(mfcc_y, x_axis='time', ax=ax[0])
ax[0].set(title='MFCC')
fig.colorbar(img1, ax=ax[0])

m_htk = librosa.feature.mfcc(y=y, sr=sr, dct_type=3)

img2 = librosa.display.specshow(m_htk, x_axis='time', ax=ax[1])
ax[1].set(title='HTK-style (dct_type=3)')
#fig.colorbar(img2, ax=[ax[1]])

def creat(audio_files):
    path = "Sons/" + "*.wav"
    audio_files = glob(path) # dossier ou se trouve les sons ---a modifier si besoin
    
    for k in audio_files:
        y,sr = librosa.load(path,sr=None)
        fft_y =np.fft.fft(y)
        n_fft = len(fft_y)
        melfb = librosa.filters.mel(sr, n_fft)
