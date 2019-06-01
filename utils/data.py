import csv
import os
import pandas as pd
import numpy as np
import cv2

def load_data(path):
    data = pd.read_csv(path)

    x_final = []
    x = []
    i=0
    y = pd.get_dummies(data['emotion']).as_matrix()
    for xseq in data['pixels'].tolist():
        x = [int(xp) for xp in xseq.split(' ')]
        x = np.asarray(x).reshape(48, 48).astype('float32')
        x_final.append([x, y[i]])
        i +=1
    x_final = np.asarray(x_final)
    np.save('dataset/training_data_raw.npy', x_final)


def prep_data():
    if not os.path.isfile('dataset/training_data_raw.npy'):
        print('loading data in files')
        load_data('dataset/fer2013.csv')

    return np.load('dataset/training_data_raw.npy')
