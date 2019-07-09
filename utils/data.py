import csv
import os
import numpy as np
import pickle
import cv2
import glob
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def generate_data_batches(files, batch_size, val_training_factor):
    # files = 'F:/emotions_detection/raw/**'
    val_training_factor = 0.7
    i = 0
    while True:
        for filename in glob.iglob(files):
            i += 1

            data = np.load(filename)

            data_x = np.array([i for i in data[0]])
            data_x = data_x[0, :, :, :]
            data_y = np.array([i for i in data[1]])

            for cbatch in range(0, data_x.shape[0], batch_size):
                batch_x = data_x[cbatch:(cbatch + batch_size),:,:,:]
                batch_y = data_y[cbatch:(cbatch + batch_size)]

                batch_x_training, batch_x_val = np.split(batch_x, [int(val_training_factor * len(batch_x))])
                batch_y_training, batch_y_val = np.split(batch_y, [int(val_training_factor * len(batch_y))])

                yield (batch_x_training, batch_y_training)

def generate_val_data_batches(files, batch_size, val_training_factor):
    # files = 'F:/emotions_detection/raw/**'
    # val_training_factor = 0.7
    i = 0
    while True:
        for filename in glob.iglob(files):
            i += 1

            data = np.load(filename)

            data_x = np.array([i for i in data[0]])
            data_x = data_x[0, :, :, :]
            data_y = np.array([i for i in data[1]])

            for cbatch in range(0, data_x.shape[0], batch_size):
                batch_x = data_x[cbatch:(cbatch + batch_size),:,:,:]
                batch_y = data_y[cbatch:(cbatch + batch_size)]

                batch_x_training, batch_x_val = np.split(batch_x, [int(val_training_factor * len(batch_x))])
                batch_y_training, batch_y_val = np.split(batch_y, [int(val_training_factor * len(batch_y))])

                yield (batch_x_val, batch_y_val)


def get_data_metric(files, batch_size, val_training_factor):
    i = 0
    batch_count = 0

    for filename in glob.iglob(files):
        i += 1

        data = np.load(filename)
        data_x = np.array([i for i in data[0]])

        for cbatch in range(0, data_x.shape[0], batch_size):
            batch_count += 1

    train_batch_count = int(val_training_factor * batch_count)
    val_batch_count = batch_count - train_batch_count

    return train_batch_count, val_batch_count

def label_categorisation(data_y):
    classes =	["agree_considered", "agree_continue", "agree_pure", "agree_reluctant", "annoyed_bothered","annoyed_rolling-eyes","arrogant", "i_did_not_hear","i_do_not_care","i_do_not_know","i_do_not_understand", "not_convinced","remember_negative","imagine_negative", "disagree_considered","Disagree_pure","Disagree_reluctant","Disbelief","Pain_felt","Pain_seen","Sad","Confused", "imagine_positiv","remember_positiv","happy_achievement","happy_laughing","happy_satiated","happy_schadenfreude","impressed", "disgust","contempt","fear_oops","fear_terror", "smiling_sardonic","smiling_triumphant","smiling_uncertain","smiling_winning","smiling_yeah-right","smiling_encouraging","smiling_endearment","smiling_flirting","smiling_sad-nostalgia"];

    y_final = []

    for i in range(len(data_y)):
        for c in range(len(classes)):
            if classes[c] in data_y[i]:
                data_y[i] = classes.index(classes[c])
                
    data_y = to_categorical(data_y, num_classes=9)

    return data_y
