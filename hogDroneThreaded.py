import cv2
from sklearn import svm
import sys
import numpy as np
from thread import start_new_thread

from os import listdir
from os.path import isfile, join

def client_thread(pos):

	for x in range(0,len(resized[0])-wSize):
		#copy = resized.copy()
		#cv2.rectangle(copy,(x,pos),(x+wSize,pos+wSize),(0,0,255),3)
		#cv2.imshow("regions",copy)
		#cv2.waitKey(1)
		roi = resized[pos:pos+wSize,x:x+wSize] #extract region of interest to check for qr code
		#copy = resized.copy()
		#cv2.imshow("thread",roi)
		#cv2.waitKey(1)
		cv2.imwrite(str(pos)+"pass.jpg",roi) #write ROI to file to pass into qr reader
		testFeats = hog.compute(cv2.imread(str(pos)+"pass.jpg"))
		test = [testFeats.flatten()]

		result = clf.predict(test)
		print(result)

			#data visualization - leave uncommented for speed				

		#cv2.rectangle(copy,(pos,i),(pos + wSize,i + wSize),(0,255,0))
		#cv2.imshow("sliding window",copy)
		#cv2.waitKey(1)	
		if (result[0] == 'drone'):
			cv2.imshow("Detected Drones",roi)
			#cv2.rectangle(testIm,(i,j),(i+wSize,j+wSize),(0,0,255),3)
			#cv2.imshow('Detected Stop Signs',testIm)
			cv2.waitKey(0)
	


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

testIm = cv2.imread(sys.argv[1])



#testIm = blur = cv2.blur(testIm,(15,15))

k = int(sys.argv[2])
print("at scale " , k)
adj = k*0.1 #move decimal one to the left to increase a tenth at a time 
resized = cv2.resize(testIm,(0,0),fx=adj,fy=adj) #resize image

cv2.imshow("circled",resized)
			
for t in range(0,len(resized),1):
	start_new_thread(client_thread,(i,))	

				
cv2.waitKey(0)




