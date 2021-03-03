# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:16:37 2021

@author: tanch
"""

from tensorflow.keras.datasets import mnist

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt

#prediction
model = keras.models.load_model('mnist_trained.h5')



img = cv2.imread('hirondelle.jpg')
plt.imshow(img)
img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
print(img.shape)

img=cv2.resize(img,(28,28))

img = img.reshape(1,28*28)
prediction = model.predict(img)

print(prediction)