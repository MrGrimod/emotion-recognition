import keras
from keras.layers import *
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from cnn_model.models import *
from utils.data import *

def main():
    x_train, y_train, x_test, y_test, input_shape = load_data('dataset/ck_dataset.pickle', True)

    model = basic_cnn(input_shape, 8)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    print('training model on raw unlabeld data \r')
    print('Shape X: ', x_train.shape, ' Shape Y: ', y_train.shape)
    print('Shape test X: ', x_test.shape, ' Shape test Y: ', y_test.shape)

    tbCallBack = keras.callbacks.TensorBoard(log_dir='./dataset/raw_training_tb', histogram_freq=0, write_graph=True, write_images=True)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=30, epochs=200, callbacks=[tbCallBack])
    model.save_weights('dataset/train_raw_weights.h5')

if __name__ == "__main__":
    main()
