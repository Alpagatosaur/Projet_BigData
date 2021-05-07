# -*- coding: utf-8 -*-
"""
Created on Tue May  4 18:49:17 2021

@author: tanch
"""

"""
TEST
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
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.array(img)/255
X_test = img.reshape(1, 100, 100, 1)
y_test = 0
Prediction = new_model.predict(X_test,y_test)
TabPred = []
for elt in Prediction[0]:
    if elt > 0.8 :
        TabPred.append(1)
    else:
        TabPred.append(0)

print("AUTRE  /   HIBOU OU CHOUETTE")
print(TabPred)



frame = cv2.imread("Output/Chouettes_Hiboux/Chouette_OU_Hibou1.png")
frame = np.array(frame)
img = cv2.resize(frame,(img_size,img_size,))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.array(img)/255
X_test = img.reshape(1, 100, 100, 1)
y_test = 1
Prediction = new_model.predict(X_test,y_test)
TabPred = []
for elt in Prediction[0]:
    if elt > 0.8 :
        TabPred.append(1)
    else:
        TabPred.append(0)
        
print("AUTRE  /   HIBOU OU CHOUETTE")
print(TabPred)