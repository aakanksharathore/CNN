# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:50:26 2017

@author: aakanksha

First code on python to read and process videos
Uses basic image segmentation steps to identify blackbuck, this can be passed 
to machine learning algorith for further classification

"""

import numpy as np
import cv2
import sys
import pandas as pd
import tkinter as tk#Tkinter as tk
from tkinter.filedialog import askopenfilename#tkFileDialog import askopenfilename

#Open the video file which needs to be processed     
root = tk.Tk()
#get screen resolution
screen_width = int(root.winfo_screenwidth())
screen_height = int(root.winfo_screenheight())

movieName =  askopenfilename(filetypes=[("Video files","*")])

cap = cv2.VideoCapture(movieName)

# Create data frames to store x and y coords of the identified blobs, rows for each individual column for each frame
nframe =cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
df = pd.DataFrame(columns=['c_id','x_px','y_px','frame'])
steps=3000
i=0-steps
row = 0
while(cap.isOpened() & (i<(nframe-steps))):
    
  i = i + steps
  cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,i)
  ret, frame = cap.read()
   # Convert frame to grayscale
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  #Equalize image
  #gray = cv2.equalizeHist(gray)
  #remove noise
  gray = cv2.medianBlur(gray,5)
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
  #params.minInertiaRatio = 0.01
 
  # Create a detector with the parameters
  ver = (cv2.__version__).split('.')
  if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
  else : 
    detector = cv2.SimpleBlobDetector_create(params)
 
  # Detect blobs.
  keypoints = detector.detect(gray)
 
  # Draw detected blobs as red circles.
  # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
  im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
  # Show keypoints
  #imS = cv2.resize(im_with_keypoints, (screen_width,screen_height)) 
  #cv2.imshow("Keypoints", imS)
  #if cv2.waitKey(1) & 0xFF == ord('q'):
  #  break 
 
  #Display image
  #cv2.imshow("Image", gray)
  #cv2.waitKey(0)
  
  #Display frame
  #cv2.imshow('frame',gray)


  #Write data to a csv file
  j = 0
  for keyPoint in keypoints:
    j = j + 1 
    row = row + 1
    x = keyPoint.pt[0]
    y = keyPoint.pt[1]
    df.loc[row] = [j, x, y, i]
    
  
  


cap.release()
#cv2.destroyAllWindows()

#Write output
df.to_csv('/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/training/samples.csv', index=False)





