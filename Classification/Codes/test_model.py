# -*- coding: utf-8 -*-
"""
Created on Tue May  4 18:49:17 2021

@author: tanch
"""
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf

img_size = 100
new_model = tf.keras.models.load_model("model.h5")
frame = cv2.imread("Output/Autres/Autre1.png")
frame = np.array(frame)
img = cv2.resize(frame,(img_size,img_size,))
img = np.array(img)/255
Prediction = new_model.predict(img)
print(Prediction)