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
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, Dense, Activation, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0

df = pd.DataFrame(pd.read_csv('base.csv'))
df = shuffle(df) #melange aleatoirement les lignes

train , test = train_test_split(df) 
X_train = train.loc[:,["Image_array"]]
y_train=train.loc[:,"Type"]

samples_df = train.loc[:,["Type","Image_array"]]
"""
training_percentage = 0.9
training_item_count = int(len(samples_df)*training_percentage)
validation_item_count = len(samples_df)-int(len(samples_df)*training_percentage)
training_df = samples_df[:training_item_count]
validation_df = samples_df[training_item_count:]
"""

#X = pd.DataFrame(columns=["Image_array"])

#y = pd.DataFrame(columns=["Type"])
#input_shape = (216,216, 3)


input_shape = (216,216, 3)
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
    
#model.summary()

"""
model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3), activation="relu", input_shape=(216,216, 3)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(2, activation="softmax"))

model.summary()
model.compile(loss='binary_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])
"""


model.compile(loss="binary_crossentropy", optimizer='adam')


target_size = (216,216)

train_datagen = ImageDataGenerator(
    rescale=1. / 255
)

model.fit(X_train,y_train,epochs = 10)
"""

training_batch_size = 32
validation_batch_size = 32
target_size = (216,216)

train_datagen = ImageDataGenerator(
    rescale=1. / 255
)
"""
"""
train_generator = train_datagen.flow_from_dataframe(
    dataframe = training_df,
    x_col='Image_array',
    y_col='Type',
    directory='/',
    target_size=target_size,
    batch_size=training_batch_size,
    shuffle=True,
    class_mode='categorical')


validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_dataframe(
    dataframe = validation_df,
    x_col='Image_array',
    y_col='Type',
    directory='/',
    target_size=target_size,
    shuffle=False,
    batch_size=validation_batch_size,
    class_mode='categorical')
"""
"""
history = model.fit(train_generator,
          epochs = 20, 
          validation_data=validation_generator,
#           class_weight=class_weights_dict,
          callbacks=callbacks)
"""

