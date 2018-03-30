#=============================================================
# Experimental Haar classifier
#=============================================================
# Can detect features in pictures
# Usage: Haar_test.py {path to image} {feature} {CV Folder} {scaleFactor}
# 	Feature (REQUIRED): 'eye' OR 'face'
#   CV Folder (REQUIRED): path to the OpenCV folder (Ex: '/home/unseen/anaconda3/share/OpenCV'
#	scaleFactor(OPTIONAL): value from 1.1 up to and including 1.9: adapt to alter recognition window
# Currently able to detect faces or eyes
# Adapted from example at https://www.superdatascience.com/opencv-face-detection/
#=============================================================

#=============================================================
# Set Up
import sys
import cv2 #openCV library
#import matplotlib.pyplot as plt # plotting figures
#%matplotlib inline

def convertToRGB(img): 
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#=============================================================
# Definitions

#-------------------------------------------------------------
# detect 
def detect_feature(f_cascade, colored_img, scaleFactor = 1.1):
    #just making a copy of image passed, so that passed image is not changed 
    img_copy = colored_img.copy()          

    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)          

    #let's detect multiscale (some images may be closer to camera than others) images
    features = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);          
    print('Features found: ', len(features))

    #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in features:
      cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)              

    return img_copy

#=============================================================
def main(argv):

	#loadimage 
	test = cv2.imread(sys.argv[1])
	feature = sys.argv[2]
	cvFolder = sys.argv[3]


	#load cascade classifier training files for haarcascade
	if feature ==  'face':
		haar_cascade = cv2.CascadeClassifier(cvFolder +'/haarcascades/haarcascade_frontalface_alt.xml')
	else:
		haar_cascade = cv2.CascadeClassifier(cvFolder +'/haarcascades/haarcascade_eye.xml')

	#call our function to detect faces
	if len (sys.argv) > 4:
		features_detected_img = detect_feature(haar_cascade, test, 
scaleFactor=float(sys.argv[4]))
	else:
		features_detected_img = detect_feature(haar_cascade, test)

	#convert image to RGB and show image 
	#plt.imshow(convertToRGB(features_detected_img))


if __name__ == "__main__":
	main(sys.argv)	
