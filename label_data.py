import csv
import os
import numpy as np
import dlib
import cv2
import glob
from tqdm import tqdm
from utils.data import *
from imutils import face_utils
from collections import OrderedDict

def main():
    i = 0
    for filename in glob.iglob('data/raw/**'):
        i += 1

        print(filename)

        data = np.load(filename, allow_pickle=True)

        dataX = np.array([i for i in data[0]])

        dataY = np.array([i for i in data[1]])

        featurePoints = detectFeatures(dataX)

        print('Data: ', data.shape)

        print('Data X: ', dataX.shape)

        print('Data X Feature Points: ', featurePoints.shape)

        print('Data Y: ', dataX.shape)

        finalData = np.array([dataX, featurePoints, dataY])

        np.save('data/labeled/'+str(i)+'.npy', finalData)


def detectFeatures(dataX):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

    featurePoints = []
    imgFeaturePoints = []
    print('Detecting data - using DLib')
    for i in tqdm(range(len(dataX[0]))):
        img = dataX[0][i]
        rects = detector(img, 0)
        for (i, rect) in enumerate(rects):
            shape = predictor(img, rect)
            shape = face_utils.shape_to_np(shape)
            
            imgFeaturePoints.append(shape)
        featurePoints.append(imgFeaturePoints)
        imgFeaturePoints = []
        
    return np.array(featurePoints)

def detectFeaturesCVCascade(xData):

    faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    eyeCascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
    mouthCascade = cv2.CascadeClassifier('cascades/haarcascade_mouth.xml')

    xFinal = []
    print('Detecting data - using OpenCv')
    for i in tqdm(range(len(xData))):
        faces = faceCascade.detectMultiScale(xData[i], 1.3, 5)

        eyes = eyeCascade.detectMultiScale(xData[i])
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(xData[i],(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        mouth = mouthCascade.detectMultiScale(xData[i])
        for (ex,ey,ew,eh) in mouth:
            cv2.rectangle(xData[i],(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        xFinal.append(xData[i])

    return xFinal


def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# return a tuple of (x, y, w, h)
	return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords


if __name__ == '__main__':
    main()
