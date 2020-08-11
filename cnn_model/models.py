from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Sequential
from keras.models import Model
from keras.layers import *
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D , MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras_applications.imagenet_utils import get_submodules_from_kwargs
from keras_applications.imagenet_utils import _obtain_input_shape

import keras_applications.imagenet_utils as imagenet_utils
import keras.backend as backend
import keras.models as models
import keras.layers as layers
import keras.utils as utils

import os
import keras
import cv2, numpy as np

def multipleInputDataModel(mplOut, cnnOut, mplIn, cnnIn, nOutPut):
    inputConcat = keras.layers.Concatenate()([mplOut, cnnOut])
    x = Dense(68, activation="relu")(inputConcat)
    x = Dense(nOutPut, activation="relu")(x)
    model = Model(inputs=[mplIn, cnnIn], outputs=x)

    return model
    
def mplModel(inputShape, nOutPut):
    inputT = Input(shape=inputShape)
    x = Dense(68, activation="relu")(inputT)
    x = Flatten()(x)
    x = Dense(nOutPut, activation="relu")(x)

    # model = Model(inputs=inputT, outputs=x)

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


def VGG16(input_shape, nOutPut):
    
    inputs = layers.Input(shape=input_shape)
        
    # Block 1
    x = layers.Conv2D(32, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(inputs)
    x = layers.Conv2D(32, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = Flatten()(x)

    x = layers.Dense(64, activation='relu', name='fc1')(x)
    x = layers.Dense(64, activation='relu', name='fc2')(x)
    x = layers.Dense(nOutPut, activation='relu', name='fc3')(x)
    
    # x = layers.Dense(classes, activation='softmax', name='predictions')(x)

    # Create model.
    # model = models.Model(inputs, x, name='vgg16')

    return inputs, x 