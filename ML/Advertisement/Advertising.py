# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:06:06 2019

@author: A.Kumar
"""

# Predict Advertisement price using Linear Regression

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,KFold,cross_val_score
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

datafile='C:/Users/a.kumar/Documents/LEARNING/DataScience/Advertisement/Advertising.csv'
df=pd.read_csv(datafile)

df.columns=[cols.replace(' ','_').lower() for cols in df.columns]
seed=7

X_cols=['tv','radio','newspaper']
y_cols='sales'

sns.jointplot(x=X_cols, y=y_cols, data=df)


sns.pairplot(df, X_cols, y_cols, size=1, aspect=0.7)

model=LinearRegression()

# to find the intercept and coef to get th relation between each feature and Target
y_axis=df['sales'].values.reshape(-1,1)

for cols in X_cols:
    X_axis=df[cols].values.reshape(-1,1)
    model.fit(X_axis,y_axis)
    print('Intercept and coef for feature: ',cols)
    print(model.intercept_)
    print(model.coef_)


y=np.array(df['sales'])
X=df.drop(['sales','unnamed:_0'],axis=1)
X=np.array(X)

# for all features
model.fit(X,y)
print(model.intercept_)
print(model.coef_)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=seed)
pd.DataFrame(X_train).to_csv('X_train.csv')
pd.DataFrame(X_test).to_csv('X_test.csv')


# checking the cross validation score for linear Model
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

gbr= GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,learning_rate = 0.05, loss = 'ls')
# checking the cross validation score for GradientBoostingRegressor Model
kfold = KFold(n_splits=10, random_state=seed)
cv_results = cross_val_score(gbr, X_train, y_train, cv=kfold)
cv_results_mean=cv_results.mean()
cv_results_std=cv_results.std()

gbr.fit(X_train,y_train)
preds_gbr=gbr.predict(X_test)
print(mean_absolute_error(y_test, preds_gbr))
print(mean_squared_error(y_test, preds_gbr))
print(r2_score(y_test, preds_gbr))

# checking the cross validation score using Decisiontree model
dt= DecisionTreeRegressor()

kfold = KFold(n_splits=10, random_state=seed)
cv_results = cross_val_score(dt, X_train, y_train, cv=kfold)
cv_results_mean=cv_results.mean()
cv_results_std=cv_results.std()

dt.fit(X_train,y_train)
preds_dt=dt.predict(X_test)
print(mean_absolute_error(y_test, preds_dt))
print(mean_squared_error(y_test, preds_dt))
print(r2_score(y_test, preds_dt))




