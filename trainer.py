import cv2
from sklearn import svm
import sys
import numpy as np
from sklearn.externals import joblib

from os import listdir
from os.path import isfile, join

wSize = 64

myPosPath = "./training/positive/64_64/"
myNegPath = "./training/negative/"

posFiles = [f for f in listdir(myPosPath) if isfile(join(myPosPath, f))]
negFiles = [f for f in listdir(myNegPath) if isfile(join(myNegPath, f))]


for f in posFiles:
	if not(f[-1] == "G"):
		print(f)
		posFiles.remove(f)

for f in negFiles:
	if not(f[-1] == "G"):
		print(f)
		negFiles.remove(f)

 

winSize = (wSize,wSize)
blockSize = (8,8)
blockStride = (4,4)
cellSize = (4,4)  #16,16
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
useSignedGradients = False

 
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, useSignedGradients)

sizeFeat = hog.compute(cv2.imread(myPosPath+posFiles[0]))

features = []
labels = []

print("LOADING IMAGES AND EXTRACTING HOG FEATURES")

for i in range(0,len(posFiles)):
	picPos = cv2.imread(myPosPath+posFiles[i])
	picNeg = cv2.imread(myNegPath+negFiles[i])
	#cv2.imshow("positive training image",picPos)
	#cv2.imshow("negative training image",picNeg)
	#cv2.waitKey(0)
	featsPos = hog.compute(picPos)
	featsNeg = hog.compute(picNeg)
	featsPos = featsPos.flatten()
	featsNeg = featsNeg.flatten()
	features.append(featsPos)	
	features.append(featsNeg)
	labels.append('drone')
	labels.append('not')





#features = [features.flatten()]
#labels = [labels.flatten()]	




clf = svm.SVC()
clf.kernel = 'rbf'

print("TRAINING SVM MODEL")

clf.fit(features,labels)

joblib.dump(clf,'droneModel.pkl')
