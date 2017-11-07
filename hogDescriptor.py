import cv2
from sklearn import svm
import sys

image = cv2.imread('honda1.jpg',0)
image2 = cv2.imread('honda2.jpg',0)
image3 = cv2.imread('not1.jpg',0)
image4 = cv2.imread('honda3.jpg',0)

testim1 = cv2.imread(sys.argv[1],0)
testim2 = cv2.imread('honda6.jpg',0)
testim3 = cv2.imread('not2.jpg',0)

winSize = (512,512)
blockSize = (32,32)
blockStride = (16,16)
cellSize = (16,16)  #16,16
nbins = 9 
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
useSignedGradients = False

 
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, useSignedGradients)

features1 = hog.compute(image)
features2 = hog.compute(image2)
features3 = hog.compute(image3)
features4 = hog.compute(image4)
features5 = hog.compute(testim1)

features1 = features1.flatten()
features2 = features2.flatten()
features3 = features3.flatten()
features4 = features4.flatten()
features5 = features5.flatten()

r = [features1,features2,features3,features4]
print(len(features1))
y = ['honda','honda','naww bruhh','honda']

clf = svm.SVC()
clf.kernel = 'linear'

clf.fit(r,y)

test = [features5]
print(clf.predict(test))

