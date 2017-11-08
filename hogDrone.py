import cv2
from sklearn import svm
import sys
import numpy as np

from os import listdir
from os.path import isfile, join

myPosPath = "./training/positive/200_200/"
myNegPath = "./training/negative/"

posFiles = [f for f in listdir(myPosPath) if isfile(join(myPosPath, f))]
negFiles = [f for f in listdir(myNegPath) if isfile(join(myNegPath, f))]


for f in posFiles:
	if not(f[-1] == "g"):
		posFiles.remove(f)

for f in negFiles:
	if not(f[-1] == "g"):
		negFiles.remove(f)

 

winSize = (200,200)
blockSize = (20,20)
blockStride = (10,10)
cellSize = (10,10)  #16,16
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
	labels.append('stop sign')
	labels.append('not')





#features = [features.flatten()]
#labels = [labels.flatten()]	




clf = svm.SVC()
clf.kernel = 'rbf'

print("TRAINING SVM MODEL")

clf.fit(features,labels)

testIm = cv2.imread(sys.argv[1])




for k in range (6,30): #increase scales
		print("at scale " , k)
		adj = k*0.1 #move decimal one to the left to increase a tenth at a time 
		resized = cv2.resize(testIm,(0,0),fx=adj,fy=adj) #resize image

		for j in range(0,len(resized)-200,k): #move window down

			for i in range(0,len(resized[0]) - 200,k): #move window right
				roi = resized[j:j+200,i:i+200] #extract region of interest to check for qr code
				copy = resized.copy()
				cv2.imwrite('pass.jpg',roi) #write ROI to file to pass into qr reader
				testFeats = hog.compute(cv2.imread('pass.jpg'))
				test = [testFeats.flatten()]

				result = clf.predict(test)
				#print(result)

				#data visualization - leave uncommented for speed				

				cv2.rectangle(copy,(i,j),(i + 200,j + 200),(0,255,0))
				#cv2.imshow("sliding window",copy)
				
				if (result[0] == 'stop sign'):
					cv2.imshow("image",roi)
					#cv2.rectangle(testIm,(i,j),(i+64,j+64),(0,0,255),3)
					#cv2.imshow('Detected Stop Signs',testIm)
					cv2.waitKey(0)
				
				
				 	





cv2.waitKey(0)
