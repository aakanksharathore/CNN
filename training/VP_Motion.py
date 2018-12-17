# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:50:26 2017

@author: aakanksha

First code on python to read and process videos
Uses basic image subtraction steps to identify blackbuck, this can be passed 
to machine learning algorith for further classification

"""

import numpy as np
import cv2
import sys
import pandas as pd

cap = cv2.VideoCapture('/home/aakanksha/Documents/Backup/Phd/Analysis/Videos/ProcessedVideos/1March_eve_01_1.avi')

#Video properties
nframe =cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT) #No. of total frames in video
fps = cap.get(cv2.cv.CV_CAP_PROP_FPS) #frame rate of video
frame_step = 5
# Craete data frames to store x and y coords of the identified blobs, rows for each individual column for each frame
df = pd.DataFrame(columns=['c_id','x_px','y_px','frame'])

i=0-frame_step
row = 0
n = 30 #difference between frames for motion detection
while(cap.isOpened()):
    
 i = i + frame_step
 #frame_n = i/nframe  
 #cap.set(2,frame_n)   #read frame_n= nth frame
 cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,i)
 ret, frame = cap.read()
  # Convert frame to grayscale
 fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 #fr=cv2.bitwise_not(fr)
 #fr = cv2.medianBlur(fr,5)
  #Invert image
  #gray = cv2.bitwise_not(gray)
 
#Read i+nth frame for motion detection
 cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,i+n)   #read frame_n th frame
 ret, fr1 = cap.read()
 fr1 = cv2.cvtColor(fr1, cv2.COLOR_BGR2GRAY)
 #fr1=cv2.bitwise_not(fr1)
 #fr1 = cv2.medianBlur(fr1,5)
 
 # Subtract one frame from the other
 img = cv2.subtract(fr1,fr)  #this operator will take care of negative values
 ret, img = cv2.threshold(img,50,255,cv2.THRESH_BINARY)  #when this doesn't work, try using adaptive thresholding
 img = cv2.medianBlur(img,5)
 #cv2.imshow("Diff2",img1)
 #cv2.waitKey(0)
 img1 = cv2.bitwise_not(img)  #python takes darker pixels as blobs
 # Setup SimpleBlobDetector parameters.
 params = cv2.SimpleBlobDetector_Params() 
 # Change thresholds
 params.minThreshold = 0;
 params.maxThreshold = 256;
     
 # Filter by Area.
 params.filterByArea = True
 params.minArea = 20 
 params.maxArea = 150
    
  # Create a detector with the parameters
 ver = (cv2.__version__).split('.')
 if int(ver[0]) < 3 :
   detector = cv2.SimpleBlobDetector(params)
 else : 
   detector = cv2.SimpleBlobDetector_create(params)
 
  # Detect blobs.
 keypoints = detector.detect(img1)
 
  # Draw detected blobs as red circles.
  # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
 im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
  # Show keypoints
 cv2.imshow("Keypoints", im_with_keypoints)
  #Write data to a csv file
 j = 0
 for keyPoint in keypoints:
    j = j + 1 
    row = row + 1
    x = keyPoint.pt[0]
    y = keyPoint.pt[1]
    df.loc[row] = [j, x, y, i]
    
  
  
  #Display image
  #cv2.imshow("Image", gray_image)
  #cv2.waitKey(0)
  
  #Display frame
  #cv2.imshow('frame',gray)
 if cv2.waitKey(1) & 0xFF == ord('q'):
     break

cap.release()
cv2.destroyAllWindows()

#Write output
df.to_csv('test.csv', index=False)





