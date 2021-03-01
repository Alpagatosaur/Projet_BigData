# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 15:23:00 2021

@author: tanch
"""

#https://medium.com/@klintcho/creating-an-open-speech-recognition-dataset-for-almost-any-language-c532fb2bc0cf



# import required libraries 
#from pydub import AudioSegment 
#from pydub.playback import play 

# Import an audio file 
# Format parameter only 
# for readability 
#wav_file = AudioSegment.from_file(file = 'chouette-hulotte-chant-et-cris1.wav', format = "wav") 

#https://medium.com/@alexandro.ramr777/audio-files-to-dataset-by-feature-extraction-with-librosa-d87adafe5b64

from glob import glob
import librosa
from librosa import feature
import numpy as np


audio_files = glob('Sons/'+'*.wav')
print(f'Number of normal audios : {len(audio_files)}')


fn_list_i = [
    feature.chroma_stft,
    feature.spectral_centroid,
    feature.spectral_bandwidth,
    feature.spectral_rolloff
]
  
fn_list_ii = [
    feature.rms,
    feature.zero_crossing_rate
]

def get_feature_vector(y,sr):  
  feat_vect_i = [ np.mean(funct(y,sr)) for funct in fn_list_i]
  feat_vect_ii = [ np.mean(funct(y)) for funct in fn_list_ii]
  
  feature_vector =   feat_vect_i + feat_vect_ii  
  return feature_vector


#build the matrix with normal audios featurized
audios_feat = []
for file in audio_files:
  '''
  y is the time series array of the audio file, a 1D np.ndarray
  sr is the sampling rate, a number
  '''  
  y,sr = librosa.load(file,sr=None)   
  feature_vector = get_feature_vector(y, sr) 
  audios_feat.append(feature_vector)  
  print('.', end= " ")
  
  
  
  
  import csv

audio_files_chouettes = 'chouettes.csv'

header =[
    'chroma_stft',
    'spectral_centroid',
    'spectral_bandwidth',
    'spectral_rolloff',
    'rms',
    'zero_crossing_rate'
]

#WARNING : this overwrites the file each time. Be aware of this because feature extraction step takes time.
with open(audio_files_chouettes,'+w') as f:
  csv_writer = csv.writer(f, delimiter = ',')
  csv_writer.writerow(header)
  csv_writer.writerows(audios_feat)

  
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.DataFrame(pd.read_csv('chouettes.csv'))

columns = df.columns
print(columns)
