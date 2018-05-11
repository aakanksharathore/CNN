#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:40:58 2018

@author: aakanksha
"""

import numpy as np
import cv2
import sys
import pandas as pd
import math as m
import Tkinter

import os
#import tensorflow
import keras
import h5py
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist 
from sklearn.model_selection import train_test_split

#Open video fileimport Tkinter
from tkFileDialog import askopenfilename
#Open the video file which needs to be processed     
root = Tkinter.Tk()

#get screen resolution
screen_width = int(root.winfo_screenwidth())
screen_height = int(root.winfo_screenheight())

movieName =  askopenfilename(filetypes=[("Video files","*")])

cap = cv2.VideoCapture(movieName)
nx = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
ny = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
# Craete data frames to store x and y coords of the identified blobs, rows for each individual column for each frame
df = pd.DataFrame(columns=['c_id','x_px','y_px','frame'])

i=0
row = 0
alt = int(input("Enter height of video(integer):  "))             
# work out size of box if box if 32x32 at 100m
grabSize = int(m.ceil((100/alt)*12))
#Load model
from keras.models import load_model
bb_model = load_model("/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/CNN/bb_model.h5py")

while(cap.isOpened()):
    
  i = i + 1
  ret, frame = cap.read()
  # Convert frame to grayscale
  #grayF = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  #Equalize image
  #gray = cv2.equalizeHist(gray)
  #remove noise
  gray = cv2.medianBlur(frame,5)
  #Invert image
  gray = cv2.bitwise_not(gray)
  # Blob detection
  # Setup SimpleBlobDetector parameters.
  params = cv2.SimpleBlobDetector_Params()
 
  # Change thresholds
  params.minThreshold = 10;
  params.maxThreshold = 110;
 
  # Filter by Area.
  #params.filterByArea = False
#  params.minArea = 100
 # params.maxArea = 150
 
  # Filter by Circularity
  params.filterByCircularity = False
  #params.minCircularity = 0.1
 
  # Filter by Convexity
  params.filterByConvexity = False
  #params.minConvexity = 0.87
 
  # Filter by Inertia
  params.filterByInertia = False
 
  # Create a detector with the parameters
  ver = (cv2.__version__).split('.')
  if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
  else : 
    detector = cv2.SimpleBlobDetector_create(params)
 
  # Detect blobs.
  keypoints = detector.detect(gray)
  if len(keypoints)==0:
       continue
  #FKP = keypoints
  FKP = []
 #re-invert the image for detection
  #gray = cv2.bitwise_not(gray)
  testX = np.ndarray(shape=(len(keypoints),40,40), dtype='uint8', order='C')
  j = 0
  for keyPoint in keypoints:
    
    ix = keyPoint.pt[0]
    iy = keyPoint.pt[1]
    #Classification: here draw boxes around keypints and classify them using svmClassifier
    tmpImg=grayF[max(0,int(iy-grabSize)):min(ny,int(iy+grabSize)), max(0,int(ix-grabSize)):min(nx,int(ix+grabSize))].copy()
    j = j + 1 
    tmpImg1=cv2.resize(tmpImg,(40,40))
  testX = tmpImg1.reshape(-1, 40,40, 3)
  testX = testX.astype('float32')
  testX = testX / 255.   
  pred = bb_model.predict(testX)
  Pclass = np.argmax(np.round(pred),axis=1)
  j=0
  indx=[]
  for pr in Pclass:
      if pr == 1:
          row = row + 1
          df.loc[row] = [j, keypoints[j].pt[0],keypoints[j].pt[1], i]
          FKP.append(keypoints[j])
          indx.append(j)
          j=j+1
          
      
  
  #Display image
  im_with_keypoints = cv2.drawKeypoints(frame, FKP, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
  #kcount=0
  #for k in FKP:
    
   # cv2.putText(im_with_keypoints ,str(indx[kcount]) ,((int(k.pt[0])+6, int(k.pt[1])-25)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(34,34,200),2)
    #kcount=kcount+1  
 
  # Show keypoints
  ims= cv2.resize(im_with_keypoints, (screen_width,screen_height)) 
  cv2.imshow("Keypoints", ims)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  #cv2.imwrite('test/keypoints.png',im_with_keypoints)

cap.release()
cv2.destroyAllWindows()

#Write output
#df.to_csv('Output_Demo.csv', index=False)