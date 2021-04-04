# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 21:10:50 2021

@author: tanch
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from glob import glob
import librosa
from librosa import feature
import numpy as np
import csv
import pandas as pd
import csv_With_path as fct_csv


csv_file = fct_csv.get_csv("Sons/")
df = pd.DataFrame(pd.read_csv(csv_file))
columns = df.columns
print(columns)


train , test = train_test_split(df)
scaler = MinMaxScaler()

X_train = train
y_train=train

#Entrainement model
model = LogisticRegression()

model.fit(X_train,y_train)

print(model.score(X_train,y_train))