# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 18:22:26 2021

@author: tanch
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


img_array = cv2.imread("Output/Autres/Autre1.png")
bw_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

