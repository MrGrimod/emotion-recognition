import csv
import os
import pandas as pd
import numpy as np
import cv2
from utils.data import *

def main():
    data = prep_data()
    face_detected = False
    for i in range(len(data)):
        face_detected = False

        img_raw = np.array(data[i][0], dtype='uint8')
        y = np.array(data[i][1])
        img_raw = cv2.resize(img_raw, (200,200))

        face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
        mouth_cascade = cv2.CascadeClassifier('cascades/haarcascade_mouth.xml')

        faces = face_cascade.detectMultiScale(img_raw, 1.3, 5)
        x_final = []
        for (x,y,w,h) in faces:
            face_detected = True
            #cv2.rectangle(img_raw,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = img_raw[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            mouth = mouth_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in mouth:
                cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        if face_detected:
            cv2.imwrite('imgs/'+str(i)+'.jpg', img_raw)
            x = np.asarray(img_raw).reshape(200, 200).astype('float32')
            x_final.append([x, y])

    x_final = np.asarray(x_final)
    print('detected faces in ', len(data), '/',len(x_final))
    np.save('dataset/training_data_labeld.npy', x_final)

if __name__ == '__main__':
    main()
