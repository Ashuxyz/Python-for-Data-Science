# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 14:12:05 2019

@author: A.Kumar
"""

# classification accuracy for Diabetic dataset

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_auc_score

datafile = 'C:/Users/a.kumar/Documents/LEARNING/DataScience_and_Analytics/Python-for-Data-Science/ML/Diabetic/diabetes.csv'
pima=pd.read_csv(datafile)

# define X and y
feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age']

# X is a matrix, hence we use [] to access the features we want in feature_cols
X = pima[feature_cols]

# y is a vector, hence we use dot to access 'label'
y = np.array(pima.Outcome)

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,stratify=y)

#Data standarization

scaler=StandardScaler()
X_scaled_train=scaler.fit_transform(X_train)
X_scaled_test=scaler.transform(X_test)



# LOGISTIC REGRESSION

# calculating the NULL accuracy to set the minimum accuracy level




logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred_class=logreg.predict(X_test)

print(accuracy_score(y_test, y_pred_class))


logreg=LogisticRegression()
logreg.fit(X_scaled_train,y_train)
y_pred_scaled_class=logreg.predict(X_scaled_test)

print(accuracy_score(y_test, y_pred_scaled_class))


# IMPORTANT: first argument is true values, second argument is predicted values
# this produces a 2x2 numpy array (matrix).
print(confusion_matrix(y_test, y_pred_class))

# to get a report for classification

print(classification_report(y_test, y_pred_class))

# print the first 10 predicted probabilities of class membership
logreg.predict_proba(X_test)[0:10]


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
	kfold = KFold(n_splits=10, random_state=9)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
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
    print('Performance Report for Algorithm : ',name,'is as below')
    print(score_dataset(model,X_train,X_test,y_train,y_test))



# WITH MORE FEATURES
    
X_full=pima.drop('Outcome',axis=1)
y_full=pima['Outcome'].values


X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_full,random_state=5,stratify=y)

# Scaling for training and test data

scaler=StandardScaler()
X_scaled_train_full=scaler.fit_transform(X_train_full)
X_scaled_test_full=scaler.transform(X_test_full)


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
	kfold = KFold(n_splits=10, random_state=9)
	cv_results = cross_val_score(model, X_scaled_train_full, y_train_full, cv=kfold, scoring='accuracy')
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
    print('Performance Report for Algorithm : ',name)
    score_dataset(model,X_scaled_train_full,X_scaled_test_full,y_train_full,y_test_full)



# Using Deep Neural Network
    
from keras.utils import to_categorical
#one-hot encode target column
y_full_cat=to_categorical(y)
y_train_cat = to_categorical(y_train_full)
y_test_cat= to_categorical(y_test_full)

# create model
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
n_cols=X_full.shape[1]
model.add(Dense(12, activation='relu',input_shape=(n_cols,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.summary()

# Model output shape
model.output_shape

# Model config
model.get_config()

# List all weight tensors 
model.get_weights()


#compile model using mse as a measure of model performance
model.compile(loss='binary_crossentropy',optimizer='adam',metrics = ['accuracy'])


from keras.callbacks import EarlyStopping
#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=3)


#train model
model.fit(X_train_full,y_train_cat,epochs=200, callbacks=[early_stopping_monitor])

preds = model.predict(X_test_full)




test_loss, test_acc = model.evaluate(X_test_full,y_test_cat)



