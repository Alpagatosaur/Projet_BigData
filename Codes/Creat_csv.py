# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 17:40:28 2021

@author: tanch
"""

from glob import glob
import librosa
import numpy as np
import csv
import pandas as pd
import matplotlib as plt


path = "Sons/" + "*.wav"
audio_files = glob(path) # dossier ou se trouve les sons ---a modifier si besoin


#y,sr = librosa.load(audio_files[0],sr=None)
#spec = librosa.feature.melspectrogram(y, sr)


#db_spec = librosa.power_to_db(spec, ref=np.max,)
#librosa.display.specshow(db_spec,y_axis='mel', x_axis='s', sr=sr)
#plt.colorbar();

List_spec=[]
List_spec_dB=[]
for i in audio_files:
    spect_i = []
    list_1D=[]
    name1 = i.split('Sons\\')
    name2 = name1[-1].split('.wav')
    name = name2[0]
    list_1D.append(name)
    y,sr = librosa.load(i,sr=None)
    spec = librosa.feature.melspectrogram(y, sr)
    db_spec = librosa.power_to_db(spec, ref=np.max,)
    spect_i.append(name)
    spect_i.append(y)
    spect_i.append(sr)
    spect_i.append(spec)
    spect_i.append(db_spec)
    List_spec.append(spect_i)
    List_spec_dB.append(db_spec)


audio_files_chouettes = 'hibou_base.csv'
header = ['Name_file', 'Audio_time_series', 'Sampling_rate_of_y', 
        'melspectrogram' , 'power_to_db']
with open(audio_files_chouettes,'+w') as f:
    csv_writer = csv.writer(f, delimiter = ',') #Evitez de changer delimiter RISQUE d erreur
    csv_writer.writerow(header)
    csv_writer.writerows(List_spec)
    
    
audio_files_chouettes = 'hibou_base_dB.csv'
header = ['power_to_db']
with open(audio_files_chouettes,'+w') as f:
    csv_writer = csv.writer(f, delimiter = ',') #Evitez de changer delimiter RISQUE d erreur
    csv_writer.writerow(header)
    csv_writer.writerows(List_spec_dB)