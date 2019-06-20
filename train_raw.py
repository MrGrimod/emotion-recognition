import keras
from keras.layers import *
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from cnn_model.models import *
from utils.data import *

def main():
    batch_size = 30

    model = basic_cnn((576, 768, 3), 9)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    print('training model on raw unlabeld data \r')

    tbCallBack = keras.callbacks.TensorBoard(log_dir='./dataset/raw_training_tb', histogram_freq=0, write_graph=True, write_images=True)

    data_gen = generate_data_batches(files='F:/emotions_detection/raw/**', batch_size=batch_size)

    val_data_gen =

    model.fit_generator(gen, steps_per_epoch=20, epochs=2, verbose=1)
    model.save_weights('storage/train_raw_weights.h5')

if __name__ == "__main__":
    main()
