# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 11:44:34 2021

@author: tanch
"""

from glob import glob
import librosa
import numpy as np
import csv

def get_csv(path):
    path = path + "*.wav"
    audio_files = glob(path)
    List_spec=[]
    List_spec_dB=[]
    for i in audio_files:
        spect_i = []
        y,sr = librosa.load(i,sr=None)
        spec = librosa.feature.melspectrogram(y, sr)
        db_spec = librosa.power_to_db(spec, ref=np.max,)
        spect_i.append(y)
        spect_i.append(sr)
        spect_i.append(spec)
        spect_i.append(db_spec)
        List_spec.append(spect_i)
        List_spec_dB.append(db_spec)
        
    
    audio_files_chouettes = 'BASE_hiboux.csv'
    with open(audio_files_chouettes,'+w') as f:
        csv_writer = csv.writer(f, delimiter = ',') #Evitez de changer delimiter RISQUE d erreur
        csv_writer.writerows(List_spec)
        
    return 'BASE_hiboux.csv'

