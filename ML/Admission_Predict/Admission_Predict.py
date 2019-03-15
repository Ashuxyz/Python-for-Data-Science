# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 17:08:40 2019

@author: A.Kumar
"""

# Prediction Adminssion possibility by using Linear Regression

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import seaborn as sns



datafile='C:/Users/a.kumar/Documents/LEARNING/DataScience/Admission_Predict/Admission_Predict.csv'
df=pd.read_csv(datafile)

df.columns=[cols.replace(' ','_').lower() for cols in df.columns]
seed=7

X_cols=['serial_no.','gre_score','toefl_score','university_rating','sop','lor_','cgpa','research']
y_cols='chance_of_admit_'

sns.pairplot(df, X_cols, y_cols, size=7, aspect=0.7)

model=LinearRegression()
y_axis=df['chance_of_admit_'].values.reshape(-1,1)

for cols in X_cols:
    X_axis=df[cols].values.reshape(-1,1)
    model.fit(X_axis,y_axis)
    print('Intercept and coef for feature: ',cols)
    print(model.intercept_)
    print(model.coef_)
    
    

y=np.array(df['chance_of_admit_'])
X=df.drop(['chance_of_admit_','serial_no.'],axis=1)
X=np.array(X)

# for all features together
model.fit(X,y)
print(model.intercept_)
print(model.coef_)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=seed)
pd.DataFrame(X_train).to_csv('X_train.csv')
pd.DataFrame(X_test).to_csv('X_test.csv')

# checking the cross validation score
kfold = KFold(n_splits=10, random_state=seed)
cv_results = cross_val_score(model, X_train, y_train, cv=kfold)
cv_results_mean=cv_results.mean()
cv_results_std=cv_results.std()

# Model testing and performance analysi
model.fit(X_train,y_train)
preds=model.predict(X_test)
pd.DataFrame(preds).to_csv('preds.csv')
print(mean_absolute_error(y_test, preds))
print(mean_squared_error(y_test, preds))
print(r2_score(y_test, preds))




