import csv
import os
import glob
import numpy as np
import dlib
import pickle
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer

files_chunks = 10

def main():
    i = 0
    c = 0
    x = []
    y = []

    encoder = LabelBinarizer()
    for filename in glob.iglob('../storage/dataset/**', recursive=True):
        if os.path.isfile(filename): # filter dirs
            i += 1
            complete_class = filename.split('\\')[3]
            #print(complete_class)
            y.append(str(complete_class))
            x.append(cv2.imread(filename, cv2.IMREAD_COLOR))

            if i >= 730:
                c += 1
                y = np.array(y)
                x = np.array(x)

                y = encoder.fit_transform(y)

                print('X Data: ', x.shape)
                print('Y Data: ', y.shape)

                file_loc = 'F:/emotions_detection/raw/'+str(c)+'.npy'

                data = np.array([[x], y])

                np.save(file_loc, data)

                print('Saved to', file_loc)

                i = 0
                x = []
                y = []


if __name__ == '__main__':
    main()
