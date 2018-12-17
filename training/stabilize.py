
import cv2
import numpy as np
import pandas as pd
import os
import re
import math
import time




warp_mode = cv2.MOTION_HOMOGRAPHY
number_of_iterations = 20

# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = -1e-16;

# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)


# set filenames for input and for saving the stabilized movie
inputName = 'Demo.avi'
outputName = '284STAB.AVI'
#166.7

# open the video
cap = cv2.VideoCapture(inputName)
fps = round(cap.get(cv2.CAP_PROP_FPS))

fStop = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))

S = (1920,1080)

# reduce to 12 frames a second - change number to required frame rate
ds = math.ceil(fps/12)

out = cv2.VideoWriter(outputName, cv2.VideoWriter_fourcc('M','J','P','G'), fps/ds, S, True)


  
im1_gray = np.array([])
first = np.array([])
fStop = 600
warp_matrix = np.eye(3, 3, dtype=np.float32) 
#warp_matrix = np.eye(2, 3, dtype=np.float32) 
full_warp = np.eye(3, 3, dtype=np.float32)
for tt in range(fStop):
    # Capture frame-by-frame
    _, frame = cap.read()

    if (tt%ds!=0):
        continue
    print(tt,fStop)
    if not(im1_gray.size):
        # enhance contrast in the image
        im1_gray = cv2.equalizeHist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        first = frame.copy()
    
    im2_gray =  cv2.equalizeHist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
    

    try:
        # find difference in movement between this frame and the last frame
        (cc, warp_matrix) = cv2.findTransformECC(im2_gray,im1_gray,warp_matrix, warp_mode, criteria)    
    except cv2.error as e:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
        
    # this frame becames the last frame for the next iteration
    im1_gray =im2_gray.copy()
    
    # alll moves are accumalated into a matrix
    #full_warp = np.dot(full_warp, np.vstack((warp_matrix,[0,0,1])))
    full_warp = np.dot(full_warp, warp_matrix)
    # create an empty image like the first frame
    im2_aligned = np.empty_like(frame)
    np.copyto(im2_aligned, first)
    # apply the transform so the image is aligned with the first frame and output to movie file
    #im2_aligned = cv2.warpAffine(frame, full_warp[0:2,:], (S[0],S[1]), dst=im2_aligned, flags=cv2.INTER_LINEAR  , borderMode=cv2.BORDER_TRANSPARENT)
    im2_aligned = cv2.warpPerspective(frame, full_warp, (S[0],S[1]), dst=im2_aligned, flags=cv2.INTER_LINEAR  , borderMode=cv2.BORDER_TRANSPARENT)
    out.write(im2_aligned)
    #cv2.imwrite(str(tt)+'stab.jpg',im2_aligned)

cap.release()
out.release()


