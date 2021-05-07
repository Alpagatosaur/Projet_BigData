# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:14:52 2021

@author: tanch
"""


from glob import glob
import librosa
import numpy as np
import csv
import pandas as pd
import matplotlib as plt
import sklearn
from PIL import Image


path = "Dataset/"
cathegories = ["Autres/" , "Chouettes_Hiboux/"]
#480 audio total

audio_files = []
for k in cathegories:
    pathSon = path + k + "*.wav"
    audio = glob(pathSon)
    audio_files.append(audio)
    
    
genre = "Autre"
List_csv = []
cpt = 0
for elt in audio_files:
    for son in elt:
        Ligne =[]
        nom = son.split(".wav")
        nom = nom[0].split("/")
        
        #Construction array image
        wave_data, wave_rate = librosa.load(son)
        wave_data, _ = librosa.effects.trim(wave_data)
        
        #On coupe le son pour avoir la meme duree
        song_sample = []
        db_array =[]
        sample_length = 5*wave_rate
        samples_from_file = []
        
        #The variable below is chosen mainly to create a 216x216 image
        N_mels=216
        cpt = cpt + 1
        cptNom = str(cpt)
        for idx in range(0,len(wave_data),sample_length):
            song_sample = wave_data[idx:idx+sample_length]
            if len(song_sample)>=sample_length:
                mel = librosa.feature.melspectrogram(song_sample, n_mels=N_mels)
                db = librosa.power_to_db(mel)
                normalised_db = sklearn.preprocessing.minmax_scale(db)
                db_array = (np.asarray(normalised_db)*255).astype(np.uint8)
                db_image =  Image.fromarray(np.array([db_array, db_array, db_array]).T)
                db_image.save("Output/"+genre+cptNom + ".png") # IMAGES
        
        if(len(db_array)!=0): #Si on a reussi a lire une image
            Ligne.append(nom[-1])
            Ligne.append(genre)
            Ligne.append(db_array)

            List_csv.append(Ligne)
    genre = "Chouette_OU_Hibou"
    cpt = 0
    
    
    
audio_files_chouettes = 'base.csv'
header = ['Name_file', 'Type' , 'Image_array']
with open(audio_files_chouettes,'+w') as f:
    csv_writer = csv.writer(f, delimiter = ',') #Evitez de changer delimiter RISQUE d erreur
    csv_writer.writerow(header)
    csv_writer.writerows(List_csv)