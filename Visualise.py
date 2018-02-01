#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 12:32:17 2018

@author: aakanksha
Visualize neural network layers
"""

bb_model = Sequential()
bb_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(40,40,1),padding='same'))
bb_model.add(LeakyReLU(alpha=0.1))
bb_model.add(MaxPooling2D((2, 2),padding='same'))


#Data
bb_batch = train_X[20444,:,:,:]

#Plot function
def obj_prnt(model,obj):
    obj=np.expand_dims(obj,axis=0)
    conv_obj = model.predict(obj)
    conv_obj2 = np.squeeze(obj,axis=0)
    
    print(conv_obj2.shape)
    
    conv_obj2=conv_obj2.reshape(conv_obj2.shape[:2])
    
    print(conv_obj2.shape)
    
    plt.imshow(conv_obj2)

#Visualize
obj_prnt(bb_model,bb_batch)