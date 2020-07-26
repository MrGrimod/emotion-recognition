import keras
from keras.layers import *
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from cnn_model.models import *
from utils.data import *
import calendar
import time

def main():
    epochs = 50
    batchSize = 5
    VALTrainingFactor = 0.7
    learningRate = 0.1
    files='data/raw/**'

    model = basicCNNModel((256, 256, 3), len(get_classes()))
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=learningRate), metrics=['accuracy'])

    print('training model on raw unlabeld data \r')

    tbCallBack = keras.callbacks.TensorBoard(log_dir='./storage/tensor_board/raw_training_tb_'+str(learningRate)+'_'+str(calendar.timegm(time.gmtime())), histogram_freq=0, write_graph=True, write_images=True)

    data_gen = generate_data_batches(files, batchSize, VALTrainingFactor)

    val_data_gen = generate_val_data_batches(files, batchSize, VALTrainingFactor)

    train_batch_count, val_batch_count = get_data_metric(files, batchSize, VALTrainingFactor)

    print('train_batch_count, val_batch_count: ', train_batch_count,', ', val_batch_count)

    model.fit_generator(data_gen, validation_data=val_data_gen, shuffle=True, validation_steps=val_batch_count, steps_per_epoch=train_batch_count, epochs=epochs, verbose=1, callbacks=[tbCallBack])
    model.save_weights('storage/train_raw_weights.h5')

if __name__ == "__main__":
    main()
