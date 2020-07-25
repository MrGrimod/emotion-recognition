from keras.models import Sequential
from keras.layers import *
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D , MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np

def multipleInputDataModel():
    inputConcat = concatenate([mlp.output, cnn.output])
    x = Dense(4, activation="relu")(inputConcat)
    x = Dense(1, activation="linear")(x)
    model = Model(inputs=[mlp.input, cnn.input], outputs=x)

    return model
def mplModel(dims):
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(4, activation="relu"))

    return model

def basicCNNModel(input_s, output_s):
    model = Sequential()

    #1st convolution layer
    model.add(Conv2D(104, (5, 5), activation='relu', input_shape=input_s))
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

def VGG_16(input_s, output_s):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=input_s))
    model.add(Conv2D (64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D (64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D (128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D (128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D (256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D (256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D (256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D (512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D (512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D (512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D (512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D (512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D (512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(812, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(812, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_s, activation='softmax'))

    return model
