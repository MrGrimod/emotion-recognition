from keras.models import Sequential
from keras.layers import *
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D , MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np

def multipleInputDataModel(mpl, cnn):
    inputConcat = concatenate([mlp.output, cnn.output])
    x = Dense(4, activation="relu")(inputConcat)
    x = Dense(1, activation="linear")(x)
    model = Model(inputs=[mlp.input, cnn.input], outputs=x)

    return model
    
def mplModel(inputShape):
    model = Sequential()
    model.add(Dense(8, input_dim=inputShape, activation="relu"))
    model.add(Dense(4, activation="relu"))

    return model

def basicCNNModel(inputShape):
    model = Sequential()

    #1st convolution layer
    model.add(Conv2D(104, (5, 5), activation='relu', input_shape=inputShape))
    model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

    #2nd convolution layer
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    #3rd convolution layer
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    model.add(Flatten())

    #fully connected neural networks
    model.add(Dense(4, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='relu'))

    # model.add(Dropout(0.2))
    # model.add(Dense(output_s, activation='softmax'))

    return model