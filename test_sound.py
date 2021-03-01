# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:48:46 2021

@author: tanch
"""
#https://iq.opengenus.org/introduction-to-librosa/
##_____________________________________________________________________________________________
#Intro
import numpy as np 
import matplotlib.pyplot as plt 
import librosa as lr


#Select audio file
#path ='Sons/le-cri-du-hibou-des-marais-asio-flammeus1.wav'
#path ='Sons/le-cri-du-hibou-des-marais-asio-flammeus2.wav'
#path ='Sons/chouette-hulotte-chant-et-cris1.wav'
#path ='Sons/chouette-hulotte-chant-et-cris2.wav'
path ='Sons/chouette-hulotte-chant-et-cris5.wav'



#Load audio file
y, sr = lr.core.load(path)

#Display amplitude / time
time = np.arange(0,len(y))/sr
fig, ax = plt.subplots()
ax.plot(time,y)
ax.set(xlabel='Time(s)',ylabel='sound amplitude')
plt.show()


##_____________________________________________________________________________________________

#Static tempo estimation
import numpy as np 
import matplotlib.pyplot as plt 
import librosa as lr

#Load audio file
#path ='le-cri-du-hibou-des-marais-asio-flammeus1.wav'
y, sr = lr.core.load(path)

onset_env = lr.onset.onset_strength(y, sr=sr)
tempo = lr.beat.tempo(onset_envelope=onset_env, sr=sr)
print(tempo)
tempo = np.asscalar(tempo)
# Compute 2-second windowed autocorrelation
hop_length = 512
ac = lr.autocorrelate(onset_env, 2 * sr // hop_length)
freqs = lr.tempo_frequencies(len(ac), sr=sr,hop_length=hop_length)
# Plot on a BPM axis.  We skip the first (0-lag) bin.
plt.figure(figsize=(8,4))
plt.semilogx(freqs[1:], lr.util.normalize(ac)[1:],label='Onset autocorrelation', basex=2)
plt.axvline(tempo, 0, 1, color='r', alpha=0.75, linestyle='--',label='Tempo: {:.2f} BPM'.format(tempo))
plt.xlabel('Tempo (BPM)')
plt.grid()
plt.title('Static tempo estimation')
plt.legend(frameon=True)
plt.axis('tight')
plt.show()

##_____________________________________________________________________________________________

#Mel spectrogram
import numpy as np 
import matplotlib.pyplot as plt 
import librosa as lr
import librosa.display

#Load audio file
#path ='le-cri-du-hibou-des-marais-asio-flammeus1.wav'
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