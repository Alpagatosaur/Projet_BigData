# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 19:21:58 2021

@author: tanch
"""

import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, Dense, Activation, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD

Datadirectory="Output/"
Classes= ["Autres/","Chouettes_Hiboux/"]
img_size = 216

Data_List=[]

for category in Classes:
    pathImg = Datadirectory + category
    pathImg = pathImg + "*.png"
    Img_files = glob(pathImg)
    genre = 0  
    for img in Img_files:
        Ligne =[]
        img_array = cv2.imread(img)
        new_array = cv2.resize(img_array,(img_size,img_size))
        bw_img = cv2.cvtColor(new_array, cv2.COLOR_BGR2GRAY)
        if(len(new_array) !=0):
            Ligne.append(bw_img)
            Ligne.append(genre)
            Data_List.append(Ligne)
    genre = 1
"""
Genre = 0  == Autre
Genre = 1 == Chouette ou hibou

"""
Data_List = shuffle(Data_List) # melange la liste
train , test = train_test_split(Data_List)
X_train=[]
y_train = []
for k in train:
    X_train.append(np.asarray(k[0]))
    y_train.append(np.asarray(k[1]))

#X_train = X_train/ 255
#X_train = tuple(X_train)
X_train = np.asarray(X_train)


y_train = np.asarray(y_train)

"""
X_train.shape
Out: (252, 216, 216)

"""

X_train = X_train.reshape(len(X_train),img_size*img_size)

input_shape = (len(X_train),img_size*img_size)



model = Sequential() # crée moi un réseau de neurones vides

model.add(Dense(200, activation="relu", input_shape=input_shape ))
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="softmax"))

sgd_optimizer = SGD(lr=2)

model.compile(loss="binary_crossentropy", optimizer=sgd_optimizer, metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)

#print(model.score(X_train, y_train))
