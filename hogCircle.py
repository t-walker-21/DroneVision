import cv2
from sklearn import svm
import sys
import numpy as np

image1 = cv2.imread('stopSigns/train/sign7.jpg')
image2 = cv2.imread('stopSigns/train/sign3.jpg')
image3 = cv2.imread('stopSigns/train/sign4.jpeg')
image4 = cv2.imread('stopSigns/train/sign5.jpg')


neg1 = cv2.imread('neg1.jpg')
neg2 = cv2.imread('neg2.jpg')
neg3 = cv2.imread('neg3.jpg')


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

features1 = hog.compute(image1)
features2 = hog.compute(image2)
features3 = hog.compute(image3)
features4 = hog.compute(image4)
negFeats1 = hog.compute(neg1)
negFeats2 = hog.compute(neg2)
negFeats3 = hog.compute(neg3)

y = ['sign','not','sign','sign','sign','not','not']

X = [features1.flatten(),negFeats1.flatten(),features2.flatten(),features3.flatten(),features4.flatten(),negFeats2.flatten(),negFeats3.flatten()]

print(len(X))

clf = svm.SVC()
clf.kernel = 'linear'

clf.fit(X,y)

testInput = hog.compute(neg1)

testInput = [testInput.flatten()]

print(clf.predict(testInput))

testIm = cv2.imread(sys.argv[1])


for k in range (1,25): #increase scales
		print("at scale " , k)
		adj = k*0.1 #move decimal one to the left to increase a tenth at a time 
		resized = cv2.resize(testIm,(0,0),fx=adj,fy=adj) #resize image

		for j in range(0,len(resized)-200,k*10): #move window down

			for i in range(0,len(resized[0])-200,k*10): #move window right
				roi = resized[j:j+200,i:i+200] #extract region of interest to check for qr code
				copy = resized.copy()
				#cv2.imshow('ROI',roi)
				#cv2.waitKey(1)
				cv2.imwrite('pass.jpg',roi) #write ROI to file to pass into qr reader			
				testFeats = hog.compute(cv2.imread('pass.jpg'))
				test = [testFeats.flatten()]

				result = clf.predict(test)
				#print("i see a ", result[0])
				#cv2.waitKey(1)
				#data visualization - leave uncommented for speed				

				cv2.rectangle(copy,(i,j),(i + 200,j + 200),(0,255,0))
				#cv2.imshow('picture',copy)
				#print(i,j)
				#cv2.waitKey(1)
				
				if (result == 'sign'):
					print("stop sign found")
					cv2.imshow("image",roi)
					#cv2.rectangle(testIm,(i,j),(i+200,j+200),(0,0,255),3)
					#cv2.imshow('Detected Stop Signs',testIm)
					cv2.waitKey(0)
					#print(j,i)
				#cv2.waitKey(1)
				
				 	





#cv2.moveWindow('Detected Circles',500,500)
cv2.waitKey(0)
