import keras
from keras.layers import *
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from cnn_model.models import *
from utils.data import *

def main():
    data = prep_data()

    model = basic_cnn((48, 48, 1), 7)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

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

    tbCallBack = keras.callbacks.TensorBoard(log_dir='./dataset/raw_training_tb', histogram_freq=0, write_graph=True, write_images=True)
    model.fit(x_train, y_train, batch_size=30, epochs=5, callbacks=[tbCallBack])
    model.save_weights('dataset/train_raw_weights.h5')

if __name__ == "__main__":
    main()
