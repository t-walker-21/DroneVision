
import cv2
import sys


from os import listdir
from os.path import isfile, join


files = [f for f in listdir(".") if isfile(join(".", f))]



for f in range(0,len(files)):
	test = files[f]
	if (test[0] == "I"):
		print(test)
		image = cv2.imread(test)
		resized_image = cv2.resize(image, (int(sys.argv[1]), int(sys.argv[2])))
		cv2.imwrite(test,resized_image)

