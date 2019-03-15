# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 10:22:02 2019

@author: A.Kumar
"""
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD
def create_model():
    # create model
    model=Sequential()
    model.add(Dense(12, activation='relu', input_shape=(11,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    #compile model
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

#create a classifier for use in scikit-learn
seed=7
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score

model=KerasClassifier(build_fn=create_model,epochs=150,batch_size=10)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#reducing overfitting using Dropout
model.add(Dropout(0.2))

#lift performance with learning rate schedule
sgd=SGD(lr=0.1,momentum=0.9,decay=0.0001,nesterov=False)
model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])