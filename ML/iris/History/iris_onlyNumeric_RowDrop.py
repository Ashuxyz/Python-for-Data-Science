# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 10:10:04 2019

@author: A.Kumar
"""

# iris.csv
# Load libraries
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error,classification_report,confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
#from xgboost import XGBRegressor

# Load dataset
datafile='C:/Users/a.kumar/Documents/LEARNING/Data Science/Datasets/iris.csv'
#url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pd.read_csv(datafile, names=names)

# to change the label of dataset columns without space and also in lower case
df.columns=[cols.replace(' ','_').lower() for cols in df.columns]
seed=7
scoring = 'accuracy'

# to find only Numeric feature types and working on missing (NaN) values
if (df.isnull().sum().sum()!=0):
    df_without_missing_rows_num=df.dropna(axis=0)
    y_without_missing_rows_num=np.array(df_without_missing_rows_num['class'])
    X_without_missing_rows_num=df_without_missing_rows_num.drop('class',axis=1)
    X_without_missing_rows_num=X_without_missing_rows_num.select_dtypes(exclude=['object'])
    X_without_missing_rows_num=np.array(X_without_missing_rows_num)
else:
    y_without_missing_rows_num=np.array(df['class'])
    X_without_missing_rows_num=df.drop('class',axis=1)
    X_without_missing_rows_num=X_without_missing_rows_num.select_dtypes(exclude=['object'])
    X_without_missing_rows_num=np.array(X_without_missing_rows_num)

# Dataset split function
def dataset_split(X,y,train_size,test_size,random_state):
    train_x, test_x,train_y,test_y = model_selection.train_test_split(X,y,train_size=train_size,test_size=test_size,random_state=random_state)
    return(train_x, test_x,train_y,test_y)
    
X_train,X_test,y_train,y_test= dataset_split(X_without_missing_rows_num,y_without_missing_rows_num,0.7,0.3,seed)

# Algorithm selection
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
#models.append(('LN', LinearRegression()))
#models.append(('RF', RandomForestRegressor()))
#models.append(('XGB', XGBRegressor()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Model fit,predict and performance report
def score_dataset(algorithm,X_train, X_test, y_train, y_test):
    model = algorithm
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
#    print(mean_absolute_error(y_test, preds))
    print(accuracy_score(y_test, preds))
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))

for name, model in models:
    print('Performance Report for Algorithm : ',name,'is as below\n',score_dataset(model,X_train,X_test,y_train,y_test))
    
    
    
