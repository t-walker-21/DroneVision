#my own implementation of HOG descriptor
import cv2
import numpy as np


image = cv2.imread('line.jpg',0)

image = np.float32(image) / 255

gx = cv2.Sobel(image,cv2.CV_32F,1,0,ksize=1)
gy = cv2.Sobel(image,cv2.CV_32F,0,1,ksize=1)

mag,angle = cv2.cartToPolar(gx,gy, angleInDegrees=True)



cv2.imshow('image',mag)

for i in range(0,len(angle),5):
	
	for j in range(0,len(angle),5):

		print(angle[i][j])
		temp = angle.copy()
		cv2.rectangle(temp,(i-5,j+5),(i+5,j-5),(255,0,0))
		
		if(angle[i][j] == 135):
			cv2.waitKey(0)
			
		cv2.waitKey(1)
		cv2.imshow('image',temp)


cv2.waitKey(0)
