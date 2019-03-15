# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 15:42:58 2019

@author: A.Kumar
"""

# Prediction housing price using Linear Regression

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,KFold,cross_val_score

datafile='C:/Users/a.kumar/Documents/LEARNING/DataScience/Housing/Housing.csv'
df=pd.read_csv(datafile)

df.columns=[cols.replace(' ','_').lower() for cols in df.columns]
seed=42

X_cols=['suburb','radio','newspaper']
y_cols='sales'

