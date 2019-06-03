import csv
import os
import pandas as pd
import numpy as np
import cv2
from utils.data import *

def main():
    data = prep_data()
    for i in range(len(data)):

        img_raw = np.array(data[i][0], dtype='uint8')

        img_raw = cv2.resize(img_raw, (200,200))

        face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
        mouth_cascade = cv2.CascadeClassifier('cascades/haarcascade_mouth.xml')

        faces = face_cascade.detectMultiScale(img_raw, 1.3, 5)
        for (x,y,w,h) in faces:
            #cv2.rectangle(img_raw,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = img_raw[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            mouth = mouth_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in mouth:
                cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        #cv2.imwrite('imgs/'+str(i)+'.jpg', img_raw)
if __name__ == '__main__':
    main()
