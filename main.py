#USAGE: python3 main.py

import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import argparse
import imutils

cap=cv2.VideoCapture(0)
cpt = 0

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model('mask.model')

while cap.isOpened():

	ret,frame= cap.read()
	cv2.rectangle(frame, (25,25),(500,400), (0,255,0),0)
	crop_frame=frame[25:400, 25:400]

	blur=cv2.GaussianBlur(crop_frame, (3,3),0)
	hsv=cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
	mask=cv2.inRange(hsv, np.array([2,0,0]), np.array([50,255,255]))

	kernel=np.ones((5,5))

	dilation= cv2.dilate(mask, kernel, iterations=1)
	erosion=cv2.erode(dilation, kernel, iterations=1)
	filtered=cv2.GaussianBlur(erosion, (3,3), 0)
	ret,thresh=cv2.threshold(filtered, 127, 255, 0)
	cv2.imshow("Original", frame)
	cv2.imshow("Press S to Save ", thresh)
	cv2.imwrite('temp.bmp',thresh)
	image=cv2.imread('temp.bmp')
	orig = image.copy()
	# pre-process the image for classification
	image = cv2.resize(image, (28, 28))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
		
	# classify the input image
	predictions = model.predict(image)[0]
   	 
	# build the label
	label=np.argmax(predictions)
	proba=predictions[label]
	label = "{}: {:.2f}%".format(label, proba * 100)

	# draw the label on the image
	output = imutils.resize(orig, width=400)
	
	cv2.putText(frame, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,	0.7, (0, 255, 0), 2)
	cpt +=1
	cv2.imshow('Press  ESC ',frame)
	k = cv2.waitKey(1)
	if k == 27:
		break
	elif k==ord('s') :
		cv2.imwrite("out/imag%04i.jpg" %cpt, thresh )



cap.release()
cv2.destroyAllWindows()




