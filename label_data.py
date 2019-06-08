import csv
import os
import numpy as np
import pickle
import cv2
from tqdm import tqdm
from utils.data import *

def main():
    x_train, y_train, x_test, y_test, input_shape = load_data('dataset/ck_dataset.pickle')

    x_final_train = detect_features(x_train)
    print(np.array(x_final_train).shape)

    x_final_test = detect_features(x_test)

    print(np.array(x_final_test).shape)

    with open('dataset/ck_dataset_labeld.pickle', 'wb') as f:
        pickle.dump({
            "training_data"   : [  np.array(x_final_train),  np.array(y_train)],
            "validation_data" : [ [], []],
            "test_data"       : [  np.array(x_final_test),  np.array(y_test)],
            "img_dim"         : {"width": x_final_train[0].shape[0], "height": x_final_train[0].shape[1]}
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

def detect_features(x_data):

    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier('cascades/haarcascade_mouth.xml')

    x_final = []
    print('Detecting data')
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

if __name__ == '__main__':
    main()
