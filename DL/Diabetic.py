# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 16:41:42 2019

@author: A.Kumar
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score

url='https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
names=['pregnancies','glucose','diastolic','triceps','insulin','bmi','dpf','age','diabetes']
df=pd.read_csv(url,names=names)


X=df.drop(columns=['diabetes'])
y=df['diabetes'].values

from keras.utils import to_categorical
#one-hot encode target column
y = to_categorical(df.diabetes)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)


scaler=StandardScaler()
X_scaled_train=scaler.fit_transform(X_train)
X_scaled_test=scaler.transform(X_test)

# create model
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
n_cols=X.shape[1]
model.add(Dense(250, activation='relu',input_shape=(n_cols,)))
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(2, activation='softmax'))

#compile model using mse as a measure of model performance
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])


from keras.callbacks import EarlyStopping
#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=3)


#train model
model.fit(X_scaled_train,y_train,epochs=30, callbacks=[early_stopping_monitor])

test_loss, test_acc = model.evaluate(X_scaled_test, y_test)


y_pred_test=model.predict(X_scaled_test)

scaled_threshold=0.6800000000000002

y_pred_binarised_test = (y_pred_test > scaled_threshold).astype("int")

accuracy=accuracy_score(y_pred_binarised_test,y_test)


