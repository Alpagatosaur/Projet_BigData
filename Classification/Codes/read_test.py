# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 18:15:13 2021

@author: tanch
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from brouillon.py import X,Y
import matplotlib.pyplot as plt
import cv2
import os
import random

import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, Dense, Activation, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0

Datadirectory="Output/"
Classes= ["Autres","Chouettes_Hiboux"]
img_size = 216

training_Data=[]
def create_training_Data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array,(img_size,img_size))
                training_Data.append([new_array,class_num])
            except:
                pass
create_training_Data()


df = pd.DataFrame(pd.read_csv('base.csv'))
df = shuffle(df) #melange aleatoirement les lignes

train , test = train_test_split(df) 
X_train = train.loc[:,["Image_array"]]
y_train=train.loc[:,"Type"]


input_shape = (216,216, 2)
effnet_layers = EfficientNetB0(weights=None, include_top=False, input_shape=input_shape)

for layer in effnet_layers.layers:
    layer.trainable = True

dropout_dense_layer = 0.3

model = Sequential()
model.add(effnet_layers)
    
model.add(MaxPooling2D())
model.add(Dense(120, use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout_dense_layer))

model.add(Dense(2, activation="softmax"))

model.compile(loss="binary_crossentropy", optimizer='adam')

model.fit(X_train,y_train,epochs = 5)