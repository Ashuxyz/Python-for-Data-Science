# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 08:51:49 2019

@author: A.Kumar
"""
# PREDICTING THE MAX TEMPERATURE FOR TOMORRROW  IN OUR CITY USING ONE YEAR OF PAST DATA
#-------------------------------------------------------------------------------------------- 
# import all relevant classes, modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
#ACQUIRE THE DATA IN ACCESSIBLE FORMAT
Datafile='C:/Users/a.kumar/Documents/LEARNING/DataScience/weather/Weathers.csv'
weather_df=pd.read_csv(Datafile)
# IDENTIFIES ANOMALIES/MISSING DATA
print(weather_df.info())
print(weather_df.describe())
print(weather_df.head())
print(weather_df.tail())
print(weather_df.shape)
# check if there is any NaN value in Dataset
print(weather_df.isnull().sum())
# DATA PREPARATION
# on hot encoding to convert text value in Numeric
weather_df=pd.get_dummies(weather_df) 
# just to check the data after hot encoding
print(weather_df.iloc[:5,:])
#  Target selection in the form of array
y=np.array(weather_df['actual']) 
# removing target from the column set
X=weather_df.drop('actual',axis=1) 
print(weather_df.head())
# stores remaining columns in the form of list
feature_list=list(weather_df.columns) 
# converting dataframe in the form of arrays
X=np.array(weather_df) 
#Training and Testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
# check shape of all data to make sure all ok 
print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)
#ESTABLISH A BASELINE
# The baseline predictions are the historical averages. Here get average baseline error which should be 
#reduced by predicting the model
baseline_preds = X_test[:, feature_list.index('average')]
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - y_test)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))
#TRAIN THE MODEL USING RANDOM FOREST 
rf=RandomForestRegressor(n_estimators = 1000,random_state=42)
rf.fit(X_train,y_train)
# PREDICTION USING RANDOM FOREST
predictions=rf.predict(X_test)
## Calculate the absolute errors
errors = abs(predictions - y_test)
# Print out the mean absolute error (mae) and compare against Average baseline calculated above.
# if less than above, it's all going good
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
#CALCULATE PERFORMANCE MATRICES
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
# Get numerical feature importances
importances = list(rf.feature_importances_)
print(importances)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
print(feature_importances)
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
print(feature_importances)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];