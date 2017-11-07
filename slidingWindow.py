#Tevon Walker

import cv2 #cropping functions
import qrtools #qr reader
import sys #command line args

image = cv2.imread(sys.argv[1]) #read image

windowSize = int(sys.argv[2]) #get size of sliding window

qr = qrtools.QR() #qr reader creation


def qrSearch(): #function to slide window over image at increasing levels of scale

	for k in range (1,8): #increase scales
		
		adj = k*0.1 #move decimal one to the left to increase a tenth at a time 
		resized = cv2.resize(image,(0,0),fx=adj,fy=adj) #resize image

		for j in range(0,len(resized)-windowSize,20*k): #move window down

			for i in range(0,len(resized[0]) - windowSize,20*k): #move window right
				roi = resized[j:j+windowSize,i:i+windowSize] #extract region of interest to check for qr code
				cv2.imwrite("temp.jpg",roi) #write ROI to file to pass into qr reader

				#data visualization - leave uncommented for speed				

				#cv2.rectangle(copy,(i,j),(i + 100,j + 100),(0,255,0))
				#cv2.imshow("image",roi)
				#cv2.waitKey(1)
				
				qr.decode("temp.jpg") #look for a qr code
				if not (qr.data == "NULL"): #if data was found...
					cv2.imshow("image",roi) #reveal qr code that was found in image
					return qr.data #return data and exit function 	
			

	return "NULL" #no qr code found, exit function



result = qrSearch() #look for qr data

if result == "NULL":
	print "no qr codes found"
	exit()

print result
cv2.waitKey(0)

