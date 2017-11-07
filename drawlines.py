import cv2
import numpy as np


image = np.zeros((64,64))

cv2.imwrite('blank.jpg',image)

cv2.circle(image,(32,32),25,(255,0,0),2)

#cv2.line(image,(255,50),(255,400),(255,0,0),50)

cv2.imshow('image',image)

cv2.imwrite('circle.jpg',image)
cv2.imwrite('circle1.jpg',image)
cv2.imwrite('circle2.jpg',image)

cv2.waitKey(0)

cv2.destroyAllWindows()
