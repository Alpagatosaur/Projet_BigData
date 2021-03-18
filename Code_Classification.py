# -*- coding: utf-8 -*-
"""

@author: tanch (Alpagatosaur)


"""
## -*- Les bibliotheques -*-
from glob import glob
import librosa
from librosa import feature
import numpy as np
import csv
import pandas as pd

## -*- Declaration des variables -*-

audio_files = [] # A completer avec la methode --check_audio_files--

list_1D = [] # A completer avec la methode --matrix_audio_featurized--
audios_feat = [] # A completer avec la methode --matrix_audio_featurized--


## -*- Les methodes -*-
def check_audio_files(path, audio_files):
    audio_files = glob(path)
    if( len(audio_files) > 0):
        return len(audio_files)
    return False


#Creation d un vecteur dans -matrix-
def get_feature_vector(y,sr,fn_list_i ,fn_list_ii ,fn_list_iii):
    feat_vect_i = [ np.mean(funct(y,sr)) for funct in fn_list_i]
    feat_vect_ii = [ np.mean(funct(y)) for funct in fn_list_ii]
    feat_vect_iii = [ np.mean(funct(y)) for funct in fn_list_iii]
    feature_vector =  feat_vect_i   + feat_vect_ii + feat_vect_iii
    return feature_vector



def creation_csv(nom_pour_csv):
    path = "Sons/" + "*.wav"
    audio_files = glob(path) # dossier ou se trouve les sons ---a modifier si besoin
    
#Differents outils pour traiter les sons
    fn_list_i = [feature.chroma_stft]
    fn_list_ii = [feature.melspectrogram]
    fn_list_iii = [feature.mfcc]
    
#Creation d une matrice a partir des sons traites
    for file in audio_files:
        #list_1D va rependre le nom de chaque audio
        name1 = file.split('Sons\\')
        name2 = name1[-1].split('.wav')
        name = name2[0]
        list_1D.append(name)
        y,sr = librosa.load(file,sr=None)   
        feature_vector = get_feature_vector(y, sr,fn_list_i ,fn_list_ii ,fn_list_iii)
        audios_feat.append(list_1D + feature_vector)
        list_1D.clear()
        
#Creation csv
    audio_files_chouettes = nom_pour_csv + '.csv'
    #entete
    header =[
    'Name_audio',
    'Chromagram',
    'Mel-scaled-spectrogram',
    'Mel-frequency-coefficients']
#WARNING : this overwrites the file each time. Be aware of this because feature extraction step takes time.
    with open(audio_files_chouettes,'+w') as f:
        csv_writer = csv.writer(f, delimiter = ',') #Evitez de changer delimiter RISQUE d erreur
        csv_writer.writerow(header)
        csv_writer.writerows(audios_feat)
#Check  
    df = pd.DataFrame(pd.read_csv(audio_files_chouettes))
    columns = df.columns
    lesnoms = df['Chromagram']
    print(columns)
    print(lesnoms)
    return " Fichier csv    OK "

##  -*- Appel fonction  -*-
creation_csv('tab_csv')

#Probleme import csv
df = pd.DataFrame(pd.read_csv('tab_csv'))
columns = df.columns
lesnoms = df['Chromagram']
print(columns)
print(lesnoms)