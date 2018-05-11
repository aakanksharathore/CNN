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

#Open video fileimport Tkinter
from tkinter.filedialog import askopenfilename
#Open the video file w
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
row = 0
cn=0
steps=500
alt = int(input("Enter height of video(integer):  "))             
# work out size of box if box if 32x32 at 100m
grabSize = int(m.ceil((100/alt)*12))
#Load model
from keras.models import load_model
bb_model = load_model("/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/CNN/bb_model.h5py")
keyP =round(len(np.arange(0,(screen_width-60),20))*len(np.arange(0,(screen_height-60),20)))
while(cap.isOpened()):
    
  i = i + steps
  ret, frame = cap.read()
  keyPoints=np.zeros((keyP,2))
  testX = np.ndarray(shape=(keyP,40,40,3), dtype='uint8', order='C')
  for j in np.arange(0,(screen_width-60),20):
      for k in np.arange(0,(screen_height-60),20):    
        ix=j+20
        iy=k+20
        tmpImg=frame[max(0,int(iy-grabSize)):min(ny,int(iy+grabSize)), max(0,int(ix-grabSize)):min(nx,int(ix+grabSize))].copy()
        tmpImg1=cv2.resize(tmpImg,(40,40))
        testX[cn,:,:,:]=tmpImg1
        keyPoints[cn][0]=ix
        keyPoints[cn][1]=iy
        cn = cn + 1
  testX = testX.reshape(-1, 40,40, 3)
  testX = testX.astype('float32')
  testX = testX / 255.   
  pred = bb_model.predict(testX)
  Pclass = np.argmax(np.round(pred),axis=1)
  t=0
  indx=[]
  FKP = []
  for pr in Pclass:
      if pr == 1:
          row = row + 1
          df.loc[row] = [t, keyPoints[t][0],keyPoints[t][1], i]
          FKP.append(keyPoints[t])
          indx.append(t)
      t=t+1
          
      
  
  #Display image
  im_with_keypoints = cv2.drawKeypoints(frame, FKP, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  

  #kcount=0
  for k in FKP:
    
    cv2.putText(frame ,'*' ,(int(k[0]), int(k[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(34,34,200),2)
    #kcount=kcount+1  
cv2.imwrite('/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/CNN/detections'+'.png',frame)
 
  # Show keypoints
#  ims= cv2.resize(im_with_keypoints, (screen_width,screen_height)) 
#  cv2.imshow("Keypoints", ims)
#  if cv2.waitKey(1) & 0xFF == ord('q'):
#    break
#
#  cv2.imwrite('/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/CNN/keypoints'+'.png',im_with_keypoints)

cap.release()
cv2.destroyAllWindows()

#Write output
#df.to_csv('Output_Demo.csv', index=False)

for i in range(len(testX)):
    cv2.imwrite('/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/CNN/testImages/im'+str(i)+'.png',testX[i])