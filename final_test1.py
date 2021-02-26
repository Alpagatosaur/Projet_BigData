# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 15:44:45 2021

@author: Marie
"""

import cv2
from tensorflow import keras

model = keras.models.load_model('test.h5')
img = cv2.imread('cat.jpg')
print(img.shape)

img = cv2.resize(img,(28,28))
print(img.shape)

img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
print(img.shape)

img = img.reshape(1,28,28,1)
prediction = model.predict(img)
print(prediction)
model.summary()