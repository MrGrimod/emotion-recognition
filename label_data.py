import csv
import os
import pandas as pd
import numpy as np
import pickle
import cv2
from utils.data import *

def main():
    x_train, y_train, x_test, y_test, input_shape = load_data('dataset/ck_dataset.pickle')

    # cv2.imshow('sad.jpg', x_train[50])
    # cv2.waitKey(3000)

    x_final_train = detect_features(x_train)

    x_final_test = detect_features(x_test)

    x_final = np.asarray(x_final)
    print('detected faces in ', len(data), '/',len(x_final))
    np.save('dataset/training_data_labeld.npy', x_final)

def save_data(data):
    with open(outfile, 'wb') as f:
        pickle.dump({
            "training_data"   : [ x_final_train, y_train],
            "validation_data" : [ x_final_test, y_test],
            "test_data"       : [ test_data, test_label],
            "img_dim"         : {"width": resize[0], "height": resize[1]}
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

def detect_features(x_data):
    face_detected = False
    for i in range(len(data)):
        face_detected = False

        face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
        mouth_cascade = cv2.CascadeClassifier('cascades/haarcascade_mouth.xml')

        faces = face_cascade.detectMultiScale(x_data, 1.3, 5)
        x_final = []
        for (x,y,w,h) in faces:
            face_detected = True
            #cv2.rectangle(x_data,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = x_data[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            mouth = mouth_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in mouth:
                cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        #cv2.imwrite('imgs/'+str(i)+'.jpg', x_data)
        x = np.asarray(x_data).reshape(200, 200).astype('float32')
        x_final.append([x, y])
    return x_final
if __name__ == '__main__':
    main()
