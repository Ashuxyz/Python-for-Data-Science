# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:14:20 2019

@author: A.Kumar
"""
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

fashion_mnist=keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#checking first image in training data

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# to scale the image pixel between 0 to 1

train_images=train_images/255.0
test_images=test_images/255.0

# showing first 25 images with its label

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
    
plt.show()

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(test_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[test_labels[i]])
    
plt.show()

# Building the model

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# compiling the models
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
# training for models

model.fit(train_images,train_labels,batch_size=128,epochs=5,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])


test_loss,test_acc=model.evaluate(test_images,test_labels)

print('Loss:',test_loss,'&','Accuracy:',test_acc)

pred_labels=model.predict_classes(test_images)

count=0
for i in range(10000):
    if pred_labels[i]!=test_labels[i]:
        count=count+1
        print('For index:',i,'Predicted:',pred_labels[i],'Expected:',test_labels[i])

print('Total Number of incorrect prediction is:',count)
