#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:34:29 2018

@author: aakanksha
"""

import sys,os
#sys.path="/home/aakanksha/virenv/keras/lib/python2.7/site-packages"
import cv2
import numpy as np
import tensorflow
import keras
import h5py
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist 
from sklearn.model_selection import train_test_split

# fix random seed for reproducibility
np.random.seed(7)

cls0 = '/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/training/no/'
cls1 = '/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/training/yes/'

lst0 = [name for name in os.listdir(cls0) if not name.startswith('.')] 
lst1 = [name for name in os.listdir(cls1) if not name.startswith('.')]
lst=[]
lst.extend(lst0)
lst.extend(lst1)
#Create image dataset
trainData = np.ndarray(shape=(len(lst),40,40), dtype='uint8', order='C')
targetData = np.hstack((np.zeros(len(lst0)),np.ones(len(lst1))))
#extract image data and append to matrix
i=0
for i in range(trainData.shape[0]):

    #print(i)
    if(i<len(lst0)):
      trainData[i-1,:,:]=cv2.resize(cv2.imread(cls0+lst[i],cv2.IMREAD_GRAYSCALE),(40,40))
    else:
       trainData[i-1,:,:]=cv2.resize(cv2.imread(cls1+lst[i],cv2.IMREAD_GRAYSCALE),(40,40))  
       
# Change the labels from categorical to one-hot encoding
targetH = to_categorical(targetData)

#data preprocessing

train_X = trainData.reshape(-1, 40,40, 1)
train_X = train_X.astype('float32')
train_X = train_X / 255.       

#Split data foe test and validation

train_X,valid_X,train_label,valid_label = train_test_split(train_X, targetH, test_size=0.4, random_state=13)  

#import all the necessary modules required to train the model.

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU           

#use a batch size of 64 using a higher batch size of 128 or 256 is also preferable it all depends on the memory. It contributes massively to determining the learning parameters and affects the prediction accuracy. You will train the network for 20 epochs.

batch_size = 64
epochs = 10
num_classes = 2

##################################################Model####################################################################

#First layer
bb_model = Sequential()
bb_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(40,40,1),padding='same'))
bb_model.add(LeakyReLU(alpha=0.1))
bb_model.add(MaxPooling2D((2, 2),padding='same'))
bb_model.add(Dropout(0.25))
#Second layer
bb_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
bb_model.add(LeakyReLU(alpha=0.1))
bb_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
bb_model.add(Dropout(0.25))

bb_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
bb_model.add(LeakyReLU(alpha=0.1))
bb_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
bb_model.add(Dropout(0.25))
#Third layer
bb_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
bb_model.add(LeakyReLU(alpha=0.1))                  
bb_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
bb_model.add(Dropout(0.4))

bb_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
bb_model.add(LeakyReLU(alpha=0.1))                  
bb_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
bb_model.add(Dropout(0.4))
#Dense layer
bb_model.add(Flatten())
bb_model.add(Dense(128, activation='linear'))
bb_model.add(LeakyReLU(alpha=0.1))        
bb_model.add(Dropout(0.3))


bb_model.add(Dense(256, activation='linear'))
bb_model.add(LeakyReLU(alpha=0.1))        
bb_model.add(Dropout(0.3))
#Output          
bb_model.add(Dense(num_classes, activation='softmax'))    
       

#############################################################################################################################

#####################################Visualise the model########################################################

#Data
#bb_batch = train_X[6,:,:,:]
#
##Plot function
#def obj_prnt(model,obj):
#    obj=np.expand_dims(obj,axis=0)
#    conv_obj = model.predict(obj)
#    conv_obj2 = np.squeeze(obj,axis=0)
#    
#    print(conv_obj2.shape)
#    
#    conv_obj2=conv_obj2.reshape(conv_obj2.shape[:2])
#    
#    print(conv_obj2.shape)
#    
#    plt.imshow(conv_obj2)
#
##Visualize
#obj_prnt(bb_model,bb_batch)

########################################################################

#Compile the model

bb_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

bb_model.summary()

#Training

bb_train = bb_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
#Epoch 25/25
#17180/17180 [==============================] - 140s 8ms/step - loss: 0.0456 - acc: 0.9867 - val_loss: 0.0869 - val_acc: 0.9804

#Model evaluation on test set
#test_eval = bb_model.evaluate(test_X, test_Y_one_hot, verbose=0)
#print('Test loss:', test_eval[0])
#print('Test accuracy:', test_eval[1])
#Check for over-fitting by plotting training and validation-loss and accuracy
accuracy =bb_train.history['acc']
val_accuracy = bb_train.history['val_acc']
loss = bb_train.history['loss']
val_loss = bb_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#Save the modelfor future
bb_model.save("/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/CNN/bb_model.h5py")
# save the model to disk
#import pickle
#filename = 'bb_model.sav'
#pickle.dump(bb_model, open(filename, 'wb'))

#Load test data

cl0 = '/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/test/no/'
cl1 = '/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/test/yes/'

ls0 = [name for name in os.listdir(cl0) if not name.startswith('.')] 
ls1 = [name for name in os.listdir(cl1) if not name.startswith('.')]
ls=[]
ls.extend(ls0)
ls.extend(ls1)
#Create image dataset
testX = np.ndarray(shape=(len(ls),40,40), dtype='uint8', order='C')
testY = np.hstack((np.zeros(len(ls0)),np.ones(len(ls1))))
#extract image data and append to matrix
i=0
for i in range(testX.shape[0]):

    #print(i)
    if(i<len(ls0)):
      testX[i-1,:,:]=cv2.resize(cv2.imread(cl0+ls[i],cv2.IMREAD_GRAYSCALE),(40,40))
    else:
       testX[i-1,:,:]=cv2.resize(cv2.imread(cl1+ls[i],cv2.IMREAD_GRAYSCALE),(40,40))  
       
# Change the labels from categorical to one-hot encoding
testY_h = to_categorical(testY)

testX = testX.reshape(-1, 40,40, 1)
testX = testX.astype('float32')
testX = testX / 255.  

#Load model
#from keras.models import load_model
#bb_model = load_model("/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/CNN/bb_model.h5py")

#Evaluation on test set



test_eval = bb_model.evaluate(testX, testY_h, verbose=1)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

#Predict labels and test predictions
predicted_classes = bb_model.predict(testX)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

correct = np.where(predicted_classes==testY)[0]
print( "Found %d correct labels" % len(correct))
incorrect = np.where(predicted_classes!=testY)[0]
print ("Found %d incorrect labels" % len(incorrect))

