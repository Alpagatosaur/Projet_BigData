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
import tensorflow as tf


Datadirectory="Output/"
Classes= ["Autres/","Chouettes_Hiboux/"]
img_size = 100

Data_List=[]
genre = 0  
for category in Classes:
    pathImg = Datadirectory + category
    pathImg = pathImg + "*.png"
    Img_files = glob(pathImg)
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


"""
 Creation d un modele
"""
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
Out: (252, 100, 100)

"""



X_train = X_train/ 255

X_train = X_train.reshape(len(X_train), 100, 100, 1)


model_conv = Sequential()
model_conv.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(100, 100, 1)))
model_conv.add(Dropout(0.3)) #dropout
model_conv.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))


model_conv.add(Flatten())

model_conv.add(Dense(100, activation="relu"))
model_conv.add(Dense(250, activation="relu"))
model_conv.add(Dense(2, activation="softmax"))


model_conv.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])



model_conv.fit(X_train, y_train, epochs=3,batch_size=10)
model_conv.save("model.h5")

"""
OPTIMISATION

"""

converter = tf.lite.TFLiteConverter.from_keras_model(model_conv)

tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)


#model_conv.summary()




"""
        TEST
        
"""
print("TEST")
#elt = 0 # L image a choisir

X_test=[]
y_test = []
for k in test:
    X_test.append(np.asarray(k[0]))
    y_test.append(np.asarray(k[1]))

X_test = np.asarray(X_test)


y_test = np.asarray(y_test)


X_test = X_test/ 255

#X_test = X_test[elt]

#y_test = y_test[elt]

#X_test = X_test.reshape(1, 100, 100, 1)
X_test = X_test.reshape(len(X_test), 100, 100, 1)

#Prediction = model_conv.predict(X_test,y_test)
score, acc = model_conv.evaluate(X_test, y_test, batch_size=10)

print("ACCURACY :")
print(acc)


"""
print("Notre animal est")
print(y_test)

print("AUTRE  /   HIBOU OU CHOUETTE")
print(Prediction)
"""