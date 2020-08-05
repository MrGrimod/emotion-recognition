from keras.models import Sequential
from keras.models import Model
from keras.layers import *
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D , MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

import keras
import cv2, numpy as np

def multipleInputDataModel(mplOut, cnnOut, mplIn, cnnIn, nOutPut):
    inputConcat = keras.layers.Concatenate()([mplOut, cnnOut])
    x = Dense(nOutPut, activation="relu")(inputConcat)
    x = Dense(nOutPut, activation="linear")(x)
    model = Model(inputs=[mplIn, cnnIn], outputs=x)

    return model
    
def mplModel(inputShape, nOutPut):
    inputT = Input(shape=inputShape)
    x = Dense(32, activation="relu")(inputT)
    x = Flatten()(x)
    x = Dense(nOutPut, activation="relu")(x)

    model = Model(inputs=inputT, outputs=x)

    return inputT, x

def basicCNNModel(inputShape, nOutPut):
    inputT = Input(shape=inputShape)

    #1st convolution layer
    x = Conv2D(128, (3, 3), activation='relu', input_shape=inputShape)(inputT)
    x = MaxPooling2D(pool_size=(5,5), strides=(2, 2))(x)

    #2nd convolution layer
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = AveragePooling2D(pool_size=(3,3), strides=(1, 1))(x)

    #3rd convolution layer
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = AveragePooling2D(pool_size=(3,3), strides=(1, 1))(x)

    x = Flatten()(x)

    #fully connected neural networks
    x = Dense(nOutPut, activation='relu')(x)
    x = Dense(nOutPut, activation='relu')(x)
    
    model = Model(inputs=inputT, outputs=x)

    return inputT, x