#=============================================================
# Experimental Haar classifier
#=============================================================
# Can detect features in pictures
# Usage: python Haar_test.py {path to /photo folder} {feature} {CV Folder} {scaleFactor}
#   Path to /photo (REQUIRED) (Ex: /home/unseen/Documents/University/SharedTask/PAN2018/en/photo
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

import os
os.path
import csv

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
     
	# return img_copy
    return len(features)

#-------------------------------------------------------------
def all_file_content(directory_name):
    file_list = os.listdir(directory_name)
    for file_name in file_list:
        yield cv2.imread(os.path.join(directory_name, file_name))



#=============================================================
def main(argv):

	#loadimage 
	mainDirPath = sys.argv[1]
	feature = sys.argv[2]
	cvFolder = sys.argv[3]

	#-------------------------------------------------------------
	#load cascade classifier training files for haarcascade
	if feature ==  'face':
		haar_cascade = cv2.CascadeClassifier(cvFolder +'/haarcascades/haarcascade_frontalface_alt.xml')
	else:
		haar_cascade = cv2.CascadeClassifier(cvFolder +'/haarcascades/haarcascade_eye.xml')

	#-------------------------------------------------------------
	#call our function to detect features
	use_list = os.listdir(mainDirPath)
	userCountList = [["UserID", "Feature counts"]]
	for user_name in use_list:
		featureCountList = [user_name]
		for file_content in all_file_content(os.path.join(mainDirPath, user_name)):
		    featureCount = detect_feature(haar_cascade, file_content)
		    featureCountList.append(featureCount)
		userCountList.append(featureCountList)
		print ("Total features for", user_name, "is:", str(sum(featureCountList[1:])))
	
	# store results in csv file
	fileName = 'HaarFeatures.csv'
	myFile = open('HaarFeatures.csv', 'w')
	with myFile:
		writer = csv.writer(myFile)
		writer.writerows(userCountList)
	print("Results written to", fileName)

	#convert image to RGB and show image 
	#plt.imshow(convertToRGB(features_detected_img))


if __name__ == "__main__":
	main(sys.argv)	
