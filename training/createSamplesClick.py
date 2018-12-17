# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:58:41 2017

@author: aakanksha

Code to create training samples from videos by clicking on images
"""


import cv2
import numpy as np
import os,sys
import math as m
import pandas as pd
#from tkinter import filedialog
import Tkinter
from Tkinter import *
from tkFileDialog import *
import matplotlib.pyplot as plt
from random import randint

def click_bb(event, x, y,flags,param):
    # grab references to the global variables
    global ref #PtX,PtY
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
          ref = [(x,y)]
          cv2.circle(fr, ref[0],32, (0, 255, 0), 2)
    else:
          ref = None
     
#Open the video file which needs to be processed     
root = Tkinter.Tk()#Tk()
movieName =  askopenfilename(filetypes=[("Video files","*")])#filedialog.askopenfilename(filetypes=[("Video files","*")])
#movieName = '/home/aakanksha/Documents/Backup/Phd/Analysis/Videos/2March/Mor/DJI_0001.MOV'
alt = input("Enter height of video(integer):  ")   
cap = cv2.VideoCapture(movieName)
nframe =cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

box_dim = 128    
nx = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
ny = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
sz=12
 # work out size of box if box if 32x32 at 100m
#alt = 40 #adjust altitude ACCORDING TO INPUT VIDEO    
grabSize = int(m.ceil((100.0/alt)*12))
frName = "Click on blackbucks and press c when done, then click on background noise"

cv2.destroyAllWindows()
cv2.namedWindow(frName, flags =  cv2.WINDOW_NORMAL)

i=1
step = 5000     #No. of frames to skip
df_bb = pd.DataFrame(columns=['x_px','y_px'])     #Store blackbuckcoordinates
df_n = pd.DataFrame(columns=['x_px','y_px'])    #Store background aand noise
cv2.namedWindow(frName)
cv2.setMouseCallback(frName,click_bb)

while(cap.isOpened()):

    if (cv2.waitKey(1) & 0xFF == ord('q')) | i > nframe:
         break
     
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,i)
    ret, frame = cap.read()
     # Convert frame to grayscale
    fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    i= i+step
     #Capture clicks on the image, fclick on blackbuck
    df_bb=df_bb[0:0]
    ref=None
    while(1):
    
        cv2.imshow(frName,fr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            break
        elif not(not(ref)):
             df_bb = df_bb.append({'x_px':ref[0][0],'y_px':ref[0][1]},ignore_index=True)
             
    
     
    df_bb=df_bb.drop_duplicates()           #Drop duplicates from the data frame
    df_bb = df_bb.reset_index(drop=True)    #Reindexing the new data frame after dropped values
    
    
    #Click on background and noise to capture negative inputs
    df_n=df_n[0:0]
    while(1):
    
        cv2.imshow(frName,fr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            break
        elif not(not(ref)):
             df_n = df_n.append({'x_px':ref[0][0],'y_px':ref[0][1]},ignore_index=True)
             
    
     
    df_n=df_n.drop_duplicates()           #Drop duplicates from the data frame
    df_n = df_n.reset_index(drop=True)    #Reindexing the new data frame after dropped values
    

    
 


    #save training data in yes and no folder
    #Write images
    
    fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)         
    for k in range(0,(len(df_bb)-1)):
            
            ix = int(df_bb.loc[k][0])
            iy = int(df_bb.loc[k][1])
            tmpImg =  fr[max(0,iy-grabSize):min(ny,iy+grabSize), max(0,ix-grabSize):min(nx,ix+grabSize)].copy()
            
            
            cv2.imwrite('/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/test/yes/' + movieName[len(movieName)-8:len(movieName)-4] +  '_' + str(i)+'_' + str(k) + str(randint(0,100))+ '.png',cv2.resize(tmpImg,(40,40)))
            
    for k in range(0,(len(df_n)-1)):   
        
            ix1 = int(df_n.loc[k][0])
            iy1 = int(df_n.loc[k][1])
            tmpImg1 =  fr[max(0,iy1-grabSize):min(ny,iy1+grabSize), max(0,ix1-grabSize):min(nx,ix1+grabSize)].copy()
            #if tmpImg.size == 4*grabSize*grabSize:# and tmpImg[tmpImg==0].size<10 :
            
            cv2.imwrite('/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/test/no/' + movieName[len(movieName)-8:len(movieName)-4] + '_'  + str(i) + '_' +str(k) + str(randint(0,100))+ '.png',cv2.resize(tmpImg1,(40,40)))

    
    
