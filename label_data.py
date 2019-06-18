import csv
import os
import numpy as np
import dlib
import pickle
import cv2
import glob
from tqdm import tqdm
from utils.data import *
from imutils import face_utils
from collections import OrderedDict

def main():
    i = 0
    for filename in glob.iglob('F:/emotions_detection/raw/**'):
        i += 1

        print(filename)

        data = np.load(filename)

        data_x = np.array([i for i in data[0]])

        data_y = np.array([i for i in data[1]])

        data_x_f = detect_features(data_x)

        print('Data: ', data.shape)

        print('Data X: ', data_x_f.shape)

        print('Data Y: ', data_y.shape)

        data_f = np.array([[data_x_f], data_y])

        np.save('F:/emotions_detection/labeled/'+str(i)+'.npy', data_f)

def detect_features(x_data):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('storage/shape_predictor_68_face_landmarks.dat')

    x_final = []
    print('Detecting data - using DLib')
    for i in tqdm(range(len(x_data[0]))):
        img = x_data[0][i]
        rects = detector(img, 0)
        for (i, rect) in enumerate(rects):
        	shape = predictor(img, rect)
        	shape = face_utils.shape_to_np(shape)

        	(x, y, w, h) = face_utils.rect_to_bb(rect)

            #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        	for (x, y) in shape:
        		cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

        # cv2.imshow("Output", img)
        # cv2.waitKey(0)
        x_final.append(img)

    x_final = np.array(x_final)

    return x_final

def detect_features_cv_cascades(x_data):

    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier('cascades/haarcascade_mouth.xml')

    x_final = []
    print('Detecting data - using OpenCv')
    for i in tqdm(range(len(x_data))):
        faces = face_cascade.detectMultiScale(x_data[i], 1.3, 5)

        eyes = eye_cascade.detectMultiScale(x_data[i])
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(x_data[i],(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        mouth = mouth_cascade.detectMultiScale(x_data[i])
        for (ex,ey,ew,eh) in mouth:
            cv2.rectangle(x_data[i],(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        x_final.append(x_data[i])

    return x_final


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
