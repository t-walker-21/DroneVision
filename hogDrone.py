import cv2
from sklearn import svm
import sys
import numpy as np

image = cv2.imread('circle.jpg',0)
blank = cv2.imread('blank.jpg',0)

winSize = (64,64)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)  #16,16
nbins = 9 
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
useSignedGradients = False

 
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, useSignedGradients)

features = hog.compute(image)
features2 = hog.compute(blank)

y = ['circle','not']

X = [features.flatten(),features2.flatten()]

clf = svm.SVC()
clf.kernel = 'linear'

clf.fit(X,y)

test = [features.flatten()]
print(clf.predict(test))




testIm = np.zeros((512,512,3))

cv2.circle(testIm,(255,255),25,(255,0,0),2);

cv2.rectangle(testIm,(20,300),(40,320),(255,255,255),2);

cv2.circle(testIm,(255,455),25,(0,0,255),2);

cv2.circle(testIm,(455,155),20,(0,255,0),2);

cv2.circle(testIm,(455,55),20,(0,255,0),2);


cv2.circle(testIm,(45,155),20,(0,255,0),2);

cv2.circle(testIm,(455,355),20,(0,255,0),2);


for k in range (10,9,-1): #increase scales
		print("at scale " , k)
		adj = k*0.1 #move decimal one to the left to increase a tenth at a time 
		resized = cv2.resize(testIm,(0,0),fx=adj,fy=adj) #resize image

		for j in range(0,len(resized)-64,k): #move window down

			for i in range(0,len(resized[0]) - 64,k): #move window right
				roi = resized[j:j+64,i:i+64] #extract region of interest to check for qr code
				copy = resized.copy()
				cv2.imwrite('pass.jpg',roi) #write ROI to file to pass into qr reader
				testFeats = hog.compute(cv2.imread('pass.jpg'))
				test = [testFeats.flatten()]

				result = clf.predict(test)

				#data visualization - leave uncommented for speed				

				#cv2.rectangle(copy,(i,j),(i + 100,j + 100),(0,255,0))
				
				if (result == 'circle'):
					cv2.imshow("image",roi)
					cv2.rectangle(testIm,(i,j),(i+64,j+64),(0,0,255),3)
					cv2.imshow('Detected Circles',testIm)
					#cv2.waitKey(0)
					print(j,i)
				#cv2.waitKey(1)
				
				 	





cv2.moveWindow('Detected Circles',500,500)
cv2.waitKey(0)
