# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 17:09:07 2019

@author: A.Kumar
"""

# Titanic Data explore

import pandas as pd

datafile1='C:/Users/a.kumar/Documents/LEARNING/DataScience/Titanic/train.csv'
datafile2='C:/Users/a.kumar/Documents/LEARNING/DataScience/Titanic/test.csv'
data_train=pd.read_csv(datafile1)
data_test=pd.read_csv(datafile2)

y_train=data_train['Survived'].values
X_train=data_train.drop('Survived',axis=1)
X_test=data_test








