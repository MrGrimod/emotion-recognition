import csv
import os
import glob
import numpy as np
import dlib
import pickle
import cv2
import sys
from imutils import face_utils
from sklearn.model_selection import train_test_split
from tqdm import tqdm
sys.path.append("..")
from utils.data import *


def main():
    files_chunk_size = 60
    i = 0
    c = 0
    x = []
    y = []

    for filename in glob.iglob('../storage/dataset/**', recursive=True):
        if os.path.isfile(filename): # filter dirs
            i += 1
            complete_class = filename.split('\\')[3]

            face_cascade = cv2.CascadeClassifier('../storage/haarcascade_frontalface_default.xml')

            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            img = detect_face(img)
            img = cv2.resize(img, (256, 256))

            cv2.imshow('img', img)
            cv2.waitKey(0)

            y.append(str(complete_class))
            x.append(img)

            if i >= files_chunk_size:
                c += 1
                y = np.array(y)

                x = np.array(x)

                y = label_categorisation(y)

                file_loc = 'F:/emotions_detection/raw/'+str(c)+'.npy'

                data = np.array([[x], y])

                np.save(file_loc, data)

                print('Saved to', file_loc)

                i = 0
                x = []
                y = []


def detect_face(img):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../storage/shape_predictor_68_face_landmarks.dat')

    rects = detector(img, 0)
    roi_color = []
    for (i, rect) in enumerate(rects):
        shape = predictor(img, rect)
        shape = face_utils.shape_to_np(shape)

        (x, y, w, h) = face_utils.rect_to_bb(rect)

        roi_color = img[y:y+h, x:x+w]

    return roi_color

if __name__ == '__main__':
    main()
