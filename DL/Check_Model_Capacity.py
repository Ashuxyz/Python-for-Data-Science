# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 10:57:45 2019

@author: A.Kumar
"""

# generate 2d classification dataset
from sklearn.datasets.samples_generator import make_blobs
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
from numpy import where

X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)

# checking the complexity by drawing scatter plot
for class_value in range(3):
    # select indices of points with the class label
    row_ix = where(y == class_value)
    plt.scatter(X[row_ix, 0], X[row_ix, 1])

plt.show()

def create_dataset():
    X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
    y=to_categorical(y)
    n_train=500
    X_train,X_test=X[0:n_train,:],X[n_train:,:]
    y_train,y_test=y[0:n_train],y[n_train:]

    return X_train,X_test,y_train,y_test

#prepare the dataset
X_train,X_test,y_train,y_test=create_dataset()


# fit model with given number of nodes, returns test set accuracy
def evaluate_model(n_nodes,n_layers,X_train,X_test,y_train,y_test):
    
	# configure the model based on the data
    n_input, n_classes = X_train.shape[1], y_train.shape[1]
	# define model
    model = Sequential()
    for layer in range(1,n_layers):
    	model.add(Dense(n_nodes, input_dim=n_input, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(n_classes, activation='softmax'))
	# compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	# fit model on train set
    history = model.fit(X_train, y_train, epochs=100, verbose=0)
	# evaluate model on test set
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    return history, test_acc

# evaluate model and plot learning curve with given number of nodes
num_nodes = np.arange(10,11)
num_layers=np.arange(1,6)
for n_layers in num_layers:
    print('Result for Layer',n_layers)
    print('------------------')
    for n_nodes in num_nodes:
	# evaluate model with a given number of nodes
    	history, result = evaluate_model(n_nodes,n_layers,X_train, X_test, y_train, y_test)
	# summarize final test set accuracy
    	print('nodes=%d: %.3f' % (n_nodes, result))
	# plot learning curve
    	plt.plot(history.history['loss'], label=str(n_nodes))
# show the plot
    plt.legend()
    plt.show()
