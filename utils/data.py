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
    data_count = 0
    for filename in glob.iglob(files):
        i += 1

        data = np.load(filename)
        data_x = np.array([i for i in data[0]])

        for cbatch in range(0, data_x.shape[0], batch_size):
            data_count += 1
    batch_count = int(data_count/batch_size)
    train_batch_count = int(val_training_factor * batch_count)
    val_batch_count = batch_count - train_batch_count

    return train_batch_count, val_batch_count

def label_categorisation(data_y):
    agree =	["agree_considered", "agree_continue", "agree_pure", "agree_reluctant"]

    annoyed = ["annoyed_bothered","annoyed_rolling-eyes","arrogant"]

    i_did_not =	["i_did_not_hear","i_do_not_care","i_do_not_know","i_do_not_understand"]

    negative = ["not_convinced","remember_negative","imagine_negative", "disagree_considered","Disagree_pure","Disagree_reluctant","Disbelief","Pain_felt","Pain_seen","Sad","Confused"]

    positive = ["imagine_positiv","remember_positiv","happy_achievement","happy_laughing","happy_satiated","happy_schadenfreude","impressed"]

    fear = ["disgust","contempt","fear_oops","fear_terror"]

    smiling = ["smiling_sardonic","smiling_triumphant","smiling_uncertain","smiling_winning","smiling_yeah-right","smiling_encouraging","smiling_endearment","smiling_flirting","smiling_sad-nostalgia"]

    y_final = []

    for i in range(len(data_y)):
        y = data_y[i]
        if y in agree:
            data_y[i] = 0
            print('Label Encoder - Agree')
        elif y in annoyed:
            data_y[i] = 2
            print('Label Encoder - annoyed')
        elif y in i_did_not:
            data_y[i] = 3
            print('Label Encoder - i_did_not')
        elif y in negative:
            data_y[i] = 4
            print('Label Encoder - negative')
        elif y in positive:
            data_y[i] = 5
            print('Label Encoder - positive')
        elif y in fear:
            data_y[i] = 6
            print('Label Encoder - fear')
        elif y in smiling:
            data_y[i] = 7
            print('Label Encoder - smiling')
        else:
            data_y[i] = 8
            print('Label Encoder - Unknown')

    data_y = to_categorical(data_y, num_classes=9)

    return data_y
