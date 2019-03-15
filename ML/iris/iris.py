# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 12:26:57 2019

@author: A.Kumar
"""

# iris.csv using KNN

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor

datafile='C:/Users/a.kumar/Documents/LEARNING/DataScience/iris/iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pd.read_csv(datafile, names=names)

df.columns=[cols.replace(' ','_').lower() for cols in df.columns]
seed=7
scoring = 'accuracy'

y=np.array(df['class'])
X=df.drop('class',axis=1)
X=np.array(X)


# Dataset split function
X_train,X_test,y_train,y_test= train_test_split(X,y,train_size=0.7,test_size=0.3,random_state=seed)
pd.DataFrame(X_train).to_csv('X_train.csv')
pd.DataFrame(X_test).to_csv('X_test.csv')

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
#models.append(('RF', RandomForestRegressor()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = KFold(n_splits=10, random_state=seed)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

def score_dataset(algorithm,X_train, X_test, y_train, y_test):
    model = algorithm
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(accuracy_score(y_test, preds))
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))

for name, model in models:
    print('Performance Report for Algorithm : ',name,'is as below\n',score_dataset(model,X_train,X_test,y_train,y_test))