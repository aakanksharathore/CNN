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
import pickle
import math as m
import tkinter
import h5py
#Open video fileimport Tkinter
from tkinter.filedialog import askopenfilename
#Open the video file which needs to be processed     
root = tkinter.Tk()

#get screen resolution
screen_width = int(root.winfo_screenwidth())
screen_height = int(root.winfo_screenheight())

movieName =  askopenfilename(filetypes=[("Video files","*")])
cap = cv2.VideoCapture(movieName)

nframe =cap.get(cv2.CAP_PROP_FRAME_COUNT)
nx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ny = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Craete data frames to store x and y coords of the identified blobs, rows for each individual column for each frame
df = pd.DataFrame(columns=['c_id','x_px','y_px','frame'])

i=0
count=0
row = 0
steps=500
#alt = int(input("Enter height of video(integer):  "))             
# work out size of box if box if 32x32 at 100m
grabSize = 15#int(m.ceil((100/alt)*12))
#Load model
from keras.models import load_model
bb_model = load_model("/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/CNN/fish_model_N12.h5py")
#Video writer object
#out = cv2.VideoWriter('/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/CNN/Output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 1, (nx,ny))

while(cap.isOpened()):
    
  i = i + steps
  cap.set(cv2.CAP_PROP_POS_FRAMES,i)
  
  ret, frame = cap.read()
  
  if ret == False:
      break;
  # Convert frame to grayscale
  grayF = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  #Equalize image
  #grayF = cv2.equalizeHist(grayF)
  #remove noise
  #gray = cv2.medianBlur(grayF,5)
  #Invert image
  #gray = cv2.bitwise_not(gray)
  # Blob detection
  # Setup SimpleBlobDetector parameters.
  params = cv2.SimpleBlobDetector_Params()
 
  # Change thresholds
  params.minThreshold = 0;
  #params.maxThreshold = 100;
 
  # Filter by Area.
  params.filterByArea = True
#  params.minArea = 100
  params.maxArea = 500
 
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
  keypoints = detector.detect(grayF)
#  if len(keypoints)==0:
#       continue
  #FKP = keypoints
 
 #re-invert the image for detection
  #gray = cv2.bitwise_not(gray)
  testX = np.ndarray(shape=(len(keypoints),40,40,3), dtype='uint8', order='C')
  j = 0
  for keyPoint in keypoints:
    
    ix = keyPoint.pt[0]
    iy = keyPoint.pt[1]
    #Classification: here draw boxes around keypints and classify them using svmClassifier
    tmpImg=frame[max(0,int(iy-grabSize)):min(ny,int(iy+grabSize)), max(0,int(ix-grabSize)):min(nx,int(ix+grabSize))].copy()
     
    tmpImg1=cv2.resize(tmpImg,(40,40))
    testX[j,:,:,:]=tmpImg1
    j = j + 1
  if not testX.any():
      break;      
  testX = testX.reshape(-1, 40,40, 3)
  testX = testX.astype('float32')
  testX = testX / 255.   
  pred = bb_model.predict(testX)
  Pclass = np.argmax(np.round(pred),axis=1)
  j=0
  indx=[]
  FKP = []
  for pr in Pclass:
      if pr == 1:
          row = row + 1
          df.loc[row] = [j, keypoints[j].pt[0],keypoints[j].pt[1], i]
          FKP.append(keypoints[j])
          indx.append(j)
      j=j+1
          
      
  
#  #Display image
  im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  cv2.putText(im_with_keypoints,'K'+','+str(i)+', '+str(len(keypoints)),(30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,(34,34,200),2)
  im_with_detections = cv2.drawKeypoints(frame, FKP, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  cv2.putText(im_with_detections,str(i)+', '+str(len(FKP)),(30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,(34,34,200),2)
    
  cv2.imwrite('/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/CNN/fishTest/'+str(i)+'keypoints'+'.png',im_with_keypoints)
#  
  cv2.imwrite('/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/CNN/fishTest/'+str(i)+'detections'+'.png',im_with_detections)
  
  count = count+1;
### 
  #write video  
#  out.write(im_with_detections)
#  #For testing
#  im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#  kcount=0
#  for k in keypoints:
#    
#    cv2.putText(im_with_keypoints ,str(indx[kcount]) ,((int(k.pt[0])+6, int(k.pt[1])-25)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(34,34,200),2)
#    kcount=kcount+1  
            
#  kcount=0
#  for k in FKP:
#    
#    cv2.putText(im_with_detections ,str(indx[kcount]) ,((int(k.pt[0])+6, int(k.pt[1])-25)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(34,34,200),2)
#    kcount=kcount+1  
 
  # Show keypoints
#  ims= cv2.resize(im_with_keypoints, (screen_width,screen_height)) 
#  cv2.imshow("Keypoints", ims)
##  if cv2.waitKey(1) & 0xFF == ord('q'):
##    break


cap.release()
#out.release()
#cv2.destroyAllWindows()

#Write output
#df.to_csv('/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/NNoutput/'+movieName[len(movieName)-20:len(movieName)-4]+'.csv', index=False)
#write output video

