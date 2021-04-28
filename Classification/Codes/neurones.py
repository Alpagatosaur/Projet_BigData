# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 15:59:58 2021

@author: tanch
"""

import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, Dense, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator



training_df = pd.DataFrame(pd.read_csv('base.csv'))
training_df = shuffle(training_df) #melange aleatoirement les lignes
X = pd.DataFrame(columns=["Image_array"])

y = pd.DataFrame(columns=["Type"])
#input_shape = (216,216, 3)


model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3), activation="relu", input_shape=(216,216, 3)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(2, activation="softmax"))

model.summary()
model.compile(loss='binary_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])

training_batch_size = 32
validation_batch_size = 32
target_size = (216,216)

train_datagen = ImageDataGenerator(
    rescale=1. / 255
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe = training_df,
    x_col=X,
    y_col=y,
    directory='/',
    target_size=target_size,
    batch_size=training_batch_size,
    shuffle=True,
    class_mode='categorical')


validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_dataframe(
    dataframe = validation_df,
    x_col=X,
    y_col=y,
    directory='/',
    target_size=target_size,
    shuffle=False,
    batch_size=validation_batch_size,
    class_mode='categorical')
