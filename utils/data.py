import csv
import os
import numpy as np
import pickle
import cv2

def load_data(path):
    with open(path, 'rb') as pickled_dataset:
        data_obj = pickle.load(pickled_dataset)

    training_data, validation_data = data_obj['training_data'], data_obj['validation_data']

    (x_train, y_train), (x_test, y_test) = (training_data[0],training_data[1]), (validation_data[0],validation_data[1])

    img_rows, img_cols = data_obj['img_dim']['width'], data_obj['img_dim']['height']

    input_shape = (img_rows, img_cols, 1)


    return x_train, y_train, x_test, y_test, input_shape

def normalize_conv_utf8(nparray):
    return (nparray*255).astype(np.uint8)
