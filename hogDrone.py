import cv2
from sklearn import svm
import sys
import numpy as np
from sklearn.externals import joblib

from os import listdir
from os.path import isfile, join

wSize = 64

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


clf = joblib.load('droneModel.pkl')

testIm = cv2.imread(sys.argv[1])


#testIm = blur = cv2.blur(testIm,(15,15))


for k in range (int(sys.argv[2]),25): #increase scales
		print("at scale " , k)
		adj = k*0.1 #move decimal one to the left to increase a tenth at a time 
		resized = cv2.resize(testIm,(0,0),fx=adj,fy=adj) #resize image

		for j in range(0,len(resized)-wSize,k): #move window down

			for i in range(0,len(resized[0]) - wSize,k): #move window right
				roi = resized[j:j+wSize,i:i+wSize] #extract region of interest to check for qr code
				copy = resized.copy()
				#cv2.imwrite('pass.jpg',roi) #write ROI to file to pass into qr reader
				#testFeats = hog.compute(cv2.imread('pass.jpg'))
				testFeats = hog.compute(roi)
				test = [testFeats.flatten()]

				result = clf.predict(test)
				#print(result)

				#data visualization - leave uncommented for speed				

				#cv2.rectangle(copy,(i,j),(i + wSize,j + wSize),(0,255,0))
				#cv2.imshow("sliding window",copy)
			 	#cv2.waitKey(1)	
				if (result[0] == 'drone'):
					cv2.imshow("Detected Drones",roi)
					#cv2.rectangle(testIm,(i,j),(i+wSize,j+wSize),(0,0,255),3)
					
					print "detected drone"
					cv2.waitKey(0)
				
				
				 	





cv2.waitKey(0)
