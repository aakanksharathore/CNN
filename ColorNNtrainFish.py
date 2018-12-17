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
from sklearn.model_selection import train_test_split

# fix random seed for reproducibility
np.random.seed(7)
#/home/abhratanu/Documents/Fish/yes
#/home/abhratanu/Documents/Fish/no
cls0 = '/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/training/FishYN/no/'
cls1 = '/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/training/FishYN/yesC/'

lst0 = [name for name in os.listdir(cls0) if not name.startswith('.')] 
lst1 = [name for name in os.listdir(cls1) if not name.startswith('.')]
lst=[]
lst.extend(lst0)
lst.extend(lst1)
#Create image dataset
trainData = np.ndarray(shape=(len(lst),40,40,3), dtype='uint8', order='C')
targetData = np.hstack((np.zeros(len(lst0)),np.ones(len(lst1))))
#extract image data and append to matrix
i=0
for i in range(trainData.shape[0]):

    #print(i)
    if(i<len(lst0)):
      im = cv2.imread(cls0+lst[i])
    else:
      im = cv2.imread(cls1+lst[i]) 
    if(im is not None):
       trainData[i-1,:,:] = cv2.resize(im,(40,40))
# Change the labels from categorical to one-hot encoding
targetH = to_categorical(targetData)

#data preprocessing

train_X = trainData.reshape(-1, 40,40, 3)
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
epochs =30
num_classes = 2

##################################################Model####################################################################

bb_model = Sequential()

bb_model.add(Conv2D(16, kernel_size=(5, 5),activation='linear',input_shape=(40,40,3),padding='same'))
bb_model.add(LeakyReLU(alpha=0.1))
#bb_model.add(MaxPooling2D((2, 2),padding='same'))

bb_model.add(Conv2D(32, kernel_size=(5, 5),activation='linear',padding='same'))
bb_model.add(LeakyReLU(alpha=0.1))
#bb_model.add(MaxPooling2D((2, 2),padding='same'))
#bb_model.add(Dropout(0.3)) 
#bb_model.add(Conv2D(16, kernel_size=(30, 30),activation='linear',padding='same'))
#bb_model.add(LeakyReLU(alpha=0.1))
#Second layer
#bb_model.add(Conv2D(96, (3, 3), activation='linear',padding='same'))
#bb_model.add(LeakyReLU(alpha=0.1))
#bb_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
#bb_model.add(Dropout(0.25))

#Third layer
#bb_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
#bb_model.add(LeakyReLU(alpha=0.3))                  
#bb_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

#Dense layer
bb_model.add(Flatten())
bb_model.add(Dense(32, activation='linear'))
bb_model.add(LeakyReLU(alpha=0.1))        
#bb_model.add(Dropout(0.3))

#bb_model.add(Dense(128, activation='linear'))
#bb_model.add(LeakyReLU(alpha=0.1))        
 
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
#    conv_obj2=conv_5obj2.reshape(conv_obj2.shape[:2])
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


#Model evaluation on test set
#test_eval = bb_model.evaluate(test_X, test_Y_one_hot, verbose=0)
#print('Test loss:', test_eval[0])
#print('Test accuracy:', test_eval[1])
#Check for over-fitting by plotting training and validation-loss and accuracy
 
#Save the model for future
bb_model.save("/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/CNN/fish_model_N12.h5py")
# save the model to disk
#import pickle
#filename = 'bb_model.sav'
#pickle.dump(bb_model, open(filename, 'wb'))

#Load test data

cl0 = '/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/test/noC/'
cl1 = '/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/test/yesC/'

ls0 = [name for name in os.listdir(cl0) if not name.startswith('.')] 
ls1 = [name for name in os.listdir(cl1) if not name.startswith('.')]
ls=[]
ls.extend(ls0)
ls.extend(ls1)
#Create image dataset
testX = np.ndarray(shape=(len(ls),10,10,3), dtype='uint8', order='C')
testY = np.hstack((np.zeros(len(ls0)),np.ones(len(ls1))))
#extract image data and append to matrix
i=0
for i in range(testX.shape[0]):

    #print(i)
    if(i<len(ls0)):
      testX[i-1,:,:]=cv2.resize(cv2.imread(cl0+ls[i]),(40,40))
    else:
       testX[i-1,:,:]=cv2.resize(cv2.imread(cl1+ls[i]),(40,40))  
       
# Change the labels from categorical to one-hot encoding
testY_h = to_categorical(testY)

testX = testX.reshape(-1, 40,40, 3)
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

