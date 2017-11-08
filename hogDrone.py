import cv2
from sklearn import svm
import sys
import numpy as np

from os import listdir
from os.path import isfile, join

mypath = "./training/positive/64_108/"

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


for f in onlyfiles:
	if not(f[-1] == "G"):
		onlyfiles.remove(f)


"""for i in range(0,len(onlyfiles)):
	print(mypath+onlyfiles[i])
	pic = cv2.imread(mypath+onlyfiles[i])
	cv2.imshow("image" + str(i),pic)


cv2.waitKey(0)
exit()"""


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
