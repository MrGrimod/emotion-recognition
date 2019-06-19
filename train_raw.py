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
    gen = generate_batches(files='F:/emotions_detection/raw/**', batch_size=batch_size)

    #model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=200, callbacks=[tbCallBack])

    # for data_batch in gen:
    #     data_x = np.array([i for i in data_batch[0]])
    #     data_y = np.array([i for i in data_batch[1]])
    #     print(data_x.shape)
    #     print(data_y.shape)
    #     print(data_y[2])
    #     cv2.imshow("Output", data_x[0])
    #     cv2.waitKey(0)

    model.fit_generator(gen, validation_steps=4, steps_per_epoch=20, epochs=2, verbose=1)
    model.save_weights('dataset/train_raw_weights.h5')

if __name__ == "__main__":
    main()
