import csv
import os
import glob
import numpy as np
import dlib
import pickle
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def main():

    x = []
    y = []

    for filename in tqdm(glob.iglob('../storage/dataset/**', recursive=True)):
        if os.path.isfile(filename): # filter dirs
            complete_class = filename.split('\\')[3]
            #print(complete_class)
            y.append(str(complete_class))
            x.append(cv2.imread(filename, cv2.IMREAD_COLOR))

    y = np.array(y)
    x = np.array(x)
    print('X Data: ', x.shape)
    print('Y Data: ', y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    with open('F:/emotions_detection/dataset_raw.pickle', 'wb') as f:
        pickle.dump({
            "training_data"   : [ x_train, y_train],
            "validation_data" : [ x_test, y_test],
            "img_dim"         : {"width": x.shape[1], "height": x.shape[2]}
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
