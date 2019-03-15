# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:02:03 2019

@author: A.Kumar
"""

 #IMDB Movie Review 
 
from keras.datasets import imdb

(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)

max(max(sequence) for sequence in train_data)


word_index=imdb.get_word_index()
reverse_word_index=dict([(value,key) for (key,value) in word_index.items()])
decode_review=''.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])

# encoding the integer sequence into a binary matrix 
#(current training and test data is in format of list of integers)
import numpy as np
def vectorize_sequences(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1
    return results

X_train=vectorize_sequences(train_data)
X_test=vectorize_sequences(test_data)      

#vectorize the labels
y_train=np.asarray(train_labels).astype('float32')
y_test=np.asarray(test_labels).astype('float32')

#creating a model

from keras import models
from keras import layers

model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#compiling the model
from keras import optimizers,losses,metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss=losses.binary_crossentropy,metrics=[metrics.binary_accuracy])

# setting aside a validation set

x_val=X_train[:10000]
partial_X_train=X_train[10000:]

y_val=y_train[:10000]
partial_y_train=y_train[10000:]

# Training the  model



    