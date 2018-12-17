import sys,os
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

cls0 = '/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/CNN/wasptrainingdata/no/'
cls1 = '/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/CNN/wasptrainingdata/yes/'

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

train_X,valid_X,train_label,valid_label = train_test_split(train_X, targetH, test_size=0.2, random_state=13)  

#import all the necessary modules required to train the model.

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU           

#use a batch size of 64 using a higher batch size of 128 or 256 is also preferable it all depends on the memory. It contributes massively to determining the learning parameters and affects the prediction accuracy. You will train the network for 20 epochs.

batch_size = 64
epochs = 30
num_classes = 2

##################################################Model####################################################################


wasp_model = Sequential()
wasp_model.add(Conv2D(64, kernel_size=(3, 3),activation='linear',input_shape=(40,40,3),padding='same'))
wasp_model.add(LeakyReLU(alpha=0.1))
wasp_model.add(MaxPooling2D((2, 2),padding='same'))

#fourth layer
wasp_model.add(Conv2D(128, kernel_size=(3, 3),activation='linear',padding='same'))
wasp_model.add(LeakyReLU(alpha=0.1))
wasp_model.add(MaxPooling2D((2, 2),padding='same'))
wasp_model.add(Dropout(0.25))

#Second layer
wasp_model.add(Conv2D(96, (3, 3), activation='linear',padding='same'))
wasp_model.add(LeakyReLU(alpha=0.1))
wasp_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
wasp_model.add(Dropout(0.25))
#Third layer
wasp_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
wasp_model.add(LeakyReLU(alpha=0.3))                  
wasp_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
#Dense layer
wasp_model.add(Flatten())
wasp_model.add(Dense(96, activation='linear'))
wasp_model.add(LeakyReLU(alpha=0.1))        
wasp_model.add(Dropout(0.25))

wasp_model.add(Dense(128, activation='linear'))
wasp_model.add(LeakyReLU(alpha=0.1))        
wasp_model.add(Dropout(0.3))


#Output          
wasp_model.add(Dense(num_classes, activation='softmax'))    
       

#Compile the model

wasp_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

wasp_model.summary()

#Training

wasp_train = wasp_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))


#Model evaluation on test set
#test_eval = wasp_model.evaluate(test_X, test_Y_one_hot, verbose=0)
#print('Test loss:', test_eval[0])
#print('Test accuracy:', test_eval[1])
#Check for over-fitting by plotting training and validation-loss and accuracy
accuracy =wasp_train.history['acc']
val_accuracy = wasp_train.history['val_acc']
loss = wasp_train.history['loss']
val_loss = wasp_train.history['val_loss']
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

#Save the model for future
wasp_model.save("/home/dexter/Desktop/wasp/wasp_classification/2way_classification/wasp_model.h5py")
# save the model to disk
#import pickle
#filename = 'wasp_model.sav'
#pickle.dump(wasp_model, open(filename, 'wb'))

#Load test data

cl0 = '/home/dexter/Desktop/wasp/wasp_classification/2way_classification/notest/'
cl1 = '/home/dexter/Desktop/wasp/wasp_classification/2way_classification/yestest/'

ls0 = [name for name in os.listdir(cl0) if not name.startswith('.')] 
ls1 = [name for name in os.listdir(cl1) if not name.startswith('.')]
ls=[]
ls.extend(ls0)
ls.extend(ls1)
#Create image dataset
testX = np.ndarray(shape=(len(ls),40,40,3), dtype='uint8', order='C')
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





test_eval = wasp_model.evaluate(testX, testY_h, verbose=1)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

#Predict labels and test predictions
predicted_classes = wasp_model.predict(testX)
#predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

#correct = np.where(predicted_classes==testY)[0]
#print( "Found %d correct labels" % len(correct))
#incorrect = np.where(predicted_classes!=testY)[0]
#print ("Found %d incorrect labels" % len(incorrect))


  	






