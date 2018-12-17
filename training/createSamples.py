

# second pass training, picks out objects identified in motion detection and now swparate them manually into yes and on training data

import cv2
import numpy as np
import os,sys
import math as m
import pandas as pd

import Tkinter
from tkFileDialog import askopenfilename

#Open the video file which needs to be processed     
root = Tkinter.Tk()
movieName =  askopenfilename(filetypes=[("Video files","*")])

## Don't forget to provide altitude information in below variable
alt = input("Enter height of video(integer):  ") 
sz=12
# work out size of box if box if 32x32 at 100m
grabSize = int(m.ceil((100.0/alt)*sz))
#HD = os.getenv('HOME')

#MOVIEDIR = HD + '/Documents/Backup/Phd/Analysis/Videos'
#DATADIR = HD + '/Documents/Backup/Phd/Analysis/Codes'
#TRACKDIR = DATADIR #+ '/tracked/'
#LOGDIR = DATADIR + '/logs/'
#FILELIST = HD + '/Documents/Backup/Phd/Analysis/Codes/textX.csv'

#df = pd.read_csv(FILELIST)

#for index, row in df.iterrows():

#    if index!=3:
#        continue
    
#noext, ext = os.path.splitext(row.filename)   


#    trackName = TRACKDIR + '/FINAL_' + str(index) + '_' + noext + '.csv'
#    geoName = LOGDIR + '/GEOTAG_' + noext + '.csv'
#    movieName = MOVIEDIR + row.filename
#movieName = '/home/aakanksha/Documents/Backup/Phd/Analysis/Videos/ProcessedVideos/1March_eve_01_1.avi'
    

        

    # name the images after the track file name
#    path, fileonly = os.path.split(trackName)
#    noext, ext = os.path.splitext(fileonly)
    
    
#    linkedDF = pd.read_csv(trackName) 
#    geoDF = pd.read_csv(geoName) 
linkedDF = pd.read_csv('/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/training/samples.csv')

numPars = int(linkedDF['frame'].count()+1)

caribouYN = np.zeros(shape=(0,3),dtype=int)
box_dim = 128    
cap = cv2.VideoCapture(movieName)

nx = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
ny = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

frName = ' is blackbuck? y or n'
cv2.destroyAllWindows()
cv2.namedWindow(frName, flags =  cv2.WINDOW_NORMAL)
escaped = False
for i in range(numPars-1):
    if i<000:
        continue
    #sys.stdout.write('\r')
    #sys.stdout.write("blackbuck id : %d" % (i))
    #sys.stdout.flush()
    #print('caribou id: ' + str(i))
    thisPar = linkedDF.loc[i]
    #thisPar = linkedDF[linkedDF['frame']==i]
    
    if escaped == True:
        break
    fNum = thisPar['frame']
    #for _, row in thisPar.iterrows():
    ix = int(thisPar['x_px'])
    iy = int(thisPar['y_px'])
    cid  = int(thisPar['c_id'])
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,fNum)
    if cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,fNum) == False:
        continue
    _, frame = cap.read()#if thisPar.count()[0]<10:
    #  continue
    cv2.rectangle(frame, ((ix-grabSize, iy-grabSize)),((ix+grabSize, iy+grabSize)),(0,0,0),1)
    tmpImg = frame[max(0,iy-box_dim/2):min(ny,iy+box_dim/2), max(0,ix-box_dim/2):min(nx,ix+box_dim/2)]
    cv2.imshow(frName,tmpImg)
    k = cv2.waitKey()
    if k==ord('y'):
       caribouYN = np.vstack((caribouYN, [fNum,cid,1]))
       continue
    if k==ord('n'):
       caribouYN = np.vstack((caribouYN, [fNum,cid,0]))
       continue
    if k==ord('c'):
       continue
    if k==27:    # Esc key to stop
       escaped=True
       break
            
cv2.destroyAllWindows()

#Write training images

print('Writing images')
for caribou in caribouYN:
        #thisPar = linkedDF[(linkedDF['frame']==caribou[0]) & (linkedDF['c_id']==caribou[1])]
        #row2 = thisPar[thisPar['c_id']==caribou[1]]
        row2 = linkedDF[(linkedDF['frame']==caribou[0]) & (linkedDF['c_id']==caribou[1])]
        
        
        ix = int(row2['x_px'])
        iy = int(row2['y_px'])
        fNum = int(row2['frame'])
        
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,fNum)
        if cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,fNum) == False:
            continue
        _, frame = cap.read()
        # get altitude of frame
        #float(geoDF['alt'][fNum])/1000.0            
        tmpImg =  cv2.cvtColor(frame[max(0,iy-grabSize):min(ny,iy+grabSize), max(0,ix-grabSize):min(nx,ix+grabSize)].copy(), cv2.COLOR_BGR2GRAY)
        #if tmpImg.size == 4*grabSize*grabSize:# and tmpImg[tmpImg==0].size<10 :
        if caribou[2]==1: #indent if you uncomment previous if statement
            cv2.imwrite('/home/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/training/yes/' +movieName[len(movieName)-8:len(movieName)-4] +  '_' + str(fNum) +'_' + str(int(caribou[1])) + '.png',cv2.resize(tmpImg,(40,40)))
        if caribou[2]==0:
            cv2.imwrite('/home/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/training/no/' + movieName[len(movieName)-8:len(movieName)-4] +  '_' + str(fNum) + '_' + str(int(caribou[1])) + '.png',cv2.resize(tmpImg,(40,40)))
            #break
        

    
cap.release()



