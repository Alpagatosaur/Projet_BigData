# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:14:22 2021

@author: tanch
"""

from tensorflow.keras.datasets import mnist

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


#Charge le modele
data = mnist.load_data()
train, test = data
X_train, y_train = train
X_train = X_train.reshape(60000, 28*28) / 255
model = Sequential() # crée moi un réseau de neurones vides
model.add(Dense(10, activation="softmax"))
sgd_optimizer = SGD(lr=3)
model.compile(loss="sparse_categorical_crossentropy", optimizer=sgd_optimizer, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32)

model.save('model_trained.h5')