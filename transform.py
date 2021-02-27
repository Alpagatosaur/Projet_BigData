# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:28:07 2021

@author: Marie
"""

#TRANSFORMER SON EN IMAGE

import numpy as np 
import matplotlib.pyplot as plt 
import librosa as lr
import librosa.display

#Load audio file
path ='le-cri-du-hibou-des-marais-asio-flammeus1.wav'
y, sr = lr.core.load(path)

lr.feature.melspectrogram(y=y, sr=sr)

D = np.abs(lr.stft(y))**2
S = lr.feature.melspectrogram(S=D)
S = lr.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
plt.figure(figsize=(10, 4))
lr.display.specshow(lr.power_to_db(S,ref=np.max),y_axis='mel', fmax=8000,x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()