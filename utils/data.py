import csv
import os
import numpy as np
import pickle
import cv2
from keras.utils import to_categorical

def load_data(path):
    #by https://github.com/mahakal/FacialEmotionRecognition
    with open(path, 'rb') as pickled_dataset:
        data_obj = pickle.load(pickled_dataset)

    (training_data, validation_data, test_data) = data_obj['training_data'], data_obj['validation_data'], data_obj['test_data']
    (x_train, y_train), (x_test, y_test) = (training_data[0],training_data[1]), (test_data[0],test_data[1])

    img_rows, img_cols = data_obj['img_dim']['width'], data_obj['img_dim']['height']
    x_train=np.array(x_train)
    x_test=np.array(x_test)
    print(x_train.shape)
    print(x_test.shape)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    y_train = to_categorical(y_train, 8)
    y_test = to_categorical(y_test, 8)

    # normalize and convert data to utf8
    x_train = normalize_conv_utf8(x_train)
    x_test = normalize_conv_utf8(x_test)

    return x_train, y_train, x_test, y_test, input_shape

def normalize_conv_utf8(nparray):
    return (nparray*255).astype(np.uint8)
