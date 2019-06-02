from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from cnn_model.models import *
from utils.data import *

def main():
    data = prep_data()

    model = VGG_16((48, 48,1), 7)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    print('progressing data')
    x_train = []
    for i in range(len(data)):
        x_train.append(data[i][0])

    y_train = []
    for i in range(len(data)):
        y_train.append(data[i][1])

    y_train = np.array(y_train)
    x_train = np.array(x_train)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)

    print('training model on raw data \r')
    print('Shape X: ', x_train.shape, ' Shape Y: ', y_train.shape)
    model.fit(x_train, y_train, epochs=3)


if __name__ == "__main__":
    main()
