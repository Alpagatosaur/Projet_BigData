# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:38:07 2021

@author: tanch
"""

from glob import glob
import librosa
from librosa import feature
import numpy as np
import csv

#Regrouper les noms des audio files
path = "Sons/" + "*.wav"
audio_files = glob(path)
print(f'Number of normal audios : {len(audio_files)}')

#Definitions des methodes de traitements
fn_list_i = [
    feature.chroma_stft
]
  
fn_list_ii = [
    feature.zero_crossing_rate
]

#Creation d un vecteur
def get_feature_vector(y,sr):
    
  feat_vect_i = [ np.mean(funct(y,sr)) for funct in fn_list_i]

  feat_vect_ii = [ np.mean(funct(y)) for funct in fn_list_ii]
  
  feature_vector =  feat_vect_i   + feat_vect_ii
      
  return feature_vector


#build the matrix with audios featurized
audios_feat = []
for file in audio_files:
  '''
  y is the time series array of the audio file, a 1D np.ndarray
  sr is the sampling rate, a number
  '''  
  #list_1D va rependre le nom de chaque audio
  list_1D = []
  list_1D.append(file)
  y,sr = librosa.load(file,sr=None)   
  feature_vector = get_feature_vector(y, sr) 
  audios_feat.append(list_1D + feature_vector)
  list_1D.remove(file)
  print('.', end= " ")
  


#Creation d un fichier csv
audio_files_chouettes = 'Data/chouettes.csv'
#Definition des titres des colonnes
header =[
    'name_audio',
    'chroma_stft',
    'zero_crossing_rate'
]

#WARNING : this overwrites the file each time. Be aware of this because feature extraction step takes time.
with open(audio_files_chouettes,'+w') as f:
  csv_writer = csv.writer(f, delimiter = ',')
  csv_writer.writerow(header)
  csv_writer.writerows(audios_feat)



#show
import pandas as pd

df = pd.DataFrame(pd.read_csv('Data/chouettes.csv'))

columns = df.columns
lesnoms = df['chroma_stft']
print(columns)
print(lesnoms)

#https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html


#Creat img with name_audio in csv

import librosa
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
from keras import layers
from keras.models import Sequential
import warnings


cmap = plt.get_cmap('inferno')
for i in range (len(audio_files)):
    file=audio_files[i]
    name1 = audio_files[i].split('Sons\\')
    name2 = name1[-1].split('.wav')
    name_file = name2[0]
    y,sr = librosa.load(file,sr=None)
    plt.specgram(y, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
    plt.axis('off');
    str_img = 'Images_with_py/' + name_file + '.png'

    plt.savefig(str_img)

    plt.show()
    

#exemple
from tensorflow.keras.datasets import mnist 
data = mnist.load_data()
train, test = data
X_train, y_train = train
img1 = X_train[0]
import matplotlib.pyplot as plt

#type numpy.ndarray
plt.imshow(img1, cmap="gray")


#png to type(img1)
#https://stackoverflow.com/questions/31386096/importing-png-files-into-numpy
import cv2

im = cv2.imread(str_img)
print(im.shape)

#creat a List type tuple
from glob import glob
List = ()
group_img = glob('Images_with_py/'+'*.png')
for image in group_img:
    image_List = np.array(cv2.imread(image))
    List = (image_List,)
    rows,cols,colors = image_List.shape

    
    

import matplotlib.pyplot as plt

img = plt.imread(str_img)
rows,cols,colors = img.shape # gives dimensions for RGB array
img_size = rows*cols*colors
img_1D_vector = img.reshape(img_size)
# you can recover the orginal image with:
img2 = img_1D_vector.reshape(rows,cols,colors)
plt.imshow(img) # followed by 
plt.show() # to show the first image, then 
plt.imshow(img2) # followed by
plt.show() # to show you the second image.

