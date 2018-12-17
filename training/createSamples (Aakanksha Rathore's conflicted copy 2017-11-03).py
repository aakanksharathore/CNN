

# second pass training

import cv2
import numpy as np
import os,sys
import math as m
import pandas as pd

HD = os.getenv('HOME')

MOVIEDIR = HD + '/Documents/Backup/Phd/Analysis/Videos/ProcessedVideos'
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
movieName = MOVIEDIR + '/1March_eve_01_1.avi'
    

        

    # name the images after the track file name
#    path, fileonly = os.path.split(trackName)
#    noext, ext = os.path.splitext(fileonly)
    
    
#    linkedDF = pd.read_csv(trackName) 
#    geoDF = pd.read_csv(geoName) 
linkedDF = pd.read_csv('test.csv')
nx = 1920
ny = 1080

numPars = int(linkedDF['frame'].max()+1)



caribouYN = np.zeros(shape=(0,3),dtype=int)
box_dim = 128    
cap = cv2.VideoCapture(movieName)

sz=16
frName = ' is blackbuck? y or n'
cv2.destroyAllWindows()
cv2.namedWindow(frName, flags =  cv2.WINDOW_NORMAL)
escaped = False
for i in range(numPars):
    if i<000:
        continue
    #sys.stdout.write('\r')
    #sys.stdout.write("blackbuck id : %d" % (i))
    #sys.stdout.flush()
    #print('caribou id: ' + str(i))
    thisPar = linkedDF[linkedDF['frame']==i]
    if escaped == True:
        break
    fNum = i
    
    for _, row in thisPar.iterrows():
        ix = int(row['x_px'])
        iy = int(row['y_px'])
        cid  = int(row['c_id'])
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,fNum)
        if cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,fNum) == False:
            continue
        _, frame = cap.read()#if thisPar.count()[0]<10:
        #  continue
        cv2.rectangle(frame, ((int( row['x_px'])-sz, int( row['y_px'])-sz)),((int( row['x_px'])+sz, int( row['y_px'])+sz)),(0,0,0),1)
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
print('Writing images')
for caribou in caribouYN:
        thisPar = linkedDF[linkedDF['c_id']==caribou[1]]
        row2 = thisPar[thisPar['frame']==caribou[0]]
        
        
        ix = int(row2['x_px'])
        iy = int(row2['y_px'])
        fNum = int(row2['frame'])
        
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,fNum)
        if cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,fNum) == False:
            continue
        _, frame = cap.read()
        # get altitude of frame
        alt = 50 #float(geoDF['alt'][fNum])/1000.0            
        # work out size of box if box if 32x32 at 100m
        grabSize = m.ceil((100.0/alt)*10.0)
        tmpImg =  cv2.cvtColor(frame[max(0,iy-grabSize):min(ny,iy+grabSize), max(0,ix-grabSize):min(nx,ix+grabSize)].copy(), cv2.COLOR_BGR2GRAY)
        #if tmpImg.size == 4*grabSize*grabSize:# and tmpImg[tmpImg==0].size<10 :
        if caribou[2]==1: #indent if you uncomment previous if statement
            cv2.imwrite('./yes/' + movieName[0:len(movieName)-4] +  '_' + str(fNum) +'_' + str(caribou[1]) + '.png',cv2.resize(tmpImg,(40,40)))
        if caribou[2]==0:
            cv2.imwrite('./no/' + movieName[0:len(movieName)-4] +  '_' + str(fNum) + '_' + str(caribou[1]) + '.png',cv2.resize(tmpImg,(40,40)))
            #break
        

    
cap.release()



