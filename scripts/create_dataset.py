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
    in_class_list = False

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../storage/shape_predictor_68_face_landmarks.dat')
    for filename in glob.iglob('../storage/dataset/**', recursive=True):
        if os.path.isfile(filename): # filter dirs
            complete_class = filename.split('\\')[3]

            for b in range(len(get_classes())):
                if str(complete_class) == get_classes()[b]:
                    in_class_list = True

            if in_class_list == True:
                img = cv2.imread(filename, cv2.IMREAD_COLOR)
                img = detect_face(img, detector, predictor)
                if type(img) is np.ndarray:
                    i += 1
                    img = cv2.resize(img, (256, 256))

                    y.append(str(complete_class))
                    x.append(img)
                else:
                    print('Did not find any faces!')

                if i >= files_chunk_size:
                    c += 1
                    y = np.array(y)

                    x = np.array(x)

                    x_final, y_final = label_categorisation(x,y)

                    file_loc = 'F:/emotions_detection/raw/'+str(c)+'.npy'

                    print(x_final.shape)
                    print(y_final.shape)

                    if not x_final.shape[0] <= 1:
                        data = np.array([[x_final], y_final])

                        np.save(file_loc, data)

                        print('Saved to', file_loc)

                    i = 0
                    x = []
                    y = []

                    in_class_list = False
            # else:
            #     print('skipped img')

def detect_face(img, detector, predictor):
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
