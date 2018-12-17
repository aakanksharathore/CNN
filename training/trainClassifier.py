import sys, os
import numpy as np
import cv2
import pickle
import itertools

from sklearn.metrics import confusion_matrix  #added by AA to check performance
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier       
from sklearn.ensemble import GradientBoostingClassifier        #added by AA 
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score   #added by AA for cross-validation
from sklearn.cross_validation import train_test_split #added by AA for splitting test-train data

#sys.path.append('../.')

from circularHOGExtractor import circularHOGExtractor
ch = circularHOGExtractor(4,5,4) 

cls0 = '/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/training/no/'
cls1 = '/media/aakanksha/f41d5ac2-703c-4b56-a960-cd3a54f21cfb/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/training/yes/'

lst0 = [name for name in os.listdir(cls0) if not name.startswith('.')] 
lst1 = [name for name in os.listdir(cls1) if not name.startswith('.')]

nFeats = ch.getNumFields()   #the total number of fields in the feature vector.
trainData = np.zeros((len(lst0)+len(lst1),nFeats))
targetData = np.hstack((np.zeros(len(lst0)),np.ones(len(lst1))))

i = 0
for imName in lst0:
    thisIm = cv2.imread(cls0 + imName,cv2.IMREAD_GRAYSCALE)
    #thisIm = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    if thisIm.size!=1600:
        continue
    trainData[i,:] = ch.extract(thisIm)
    i = i + 1
for imName in lst1:
    thisIm = cv2.imread(cls1 + imName,cv2.IMREAD_GRAYSCALE)
    if thisIm.size!=1600:
        continue
    #thisIm = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    trainData[i,:] = ch.extract(thisIm)
    i = i + 1

#clf = svm.SVC()

#clf = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=5,min_samples_split=5),algorithm="SAMME",n_estimators=300)
clf=GradientBoostingClassifier(n_estimators=300, min_samples_split=5, min_samples_leaf=5,subsample=0.8)
clf.fit(trainData,targetData)
pickle.dump(clf, open( "/home/aakanksha/Documents/Backup/Phd/Analysis/blackbuckML/svmClassifier2.p","wb"))

#print("Number of mislabeled points out of a total %d points : %d" % (trainData.shape[0],(targetData != y_pred).sum()))
scores = cross_val_score(clf, trainData, targetData, cv=15)
scores 
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) 
y_pred = clf.predict(trainData)
cfm = confusion_matrix(targetData, y_pred, labels=[0, 1])

#Visualize the classifier
cl=DecisionTreeClassifier(min_samples_leaf=5,min_samples_split=5)
from sklearn import tree
with open("bb_classifier.txt", "w") as f:
    f = tree.export_graphviz(cl, out_file=f)