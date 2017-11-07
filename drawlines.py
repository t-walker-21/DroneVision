import cv2
import numpy as np


image = np.zeros((512,512))


cv2.line(image,(50,50),(400,400),(255,0,0),15)

cv2.imshow('image',image)


cv2.waitKey(0)

cv2.destroyAllWindows()
