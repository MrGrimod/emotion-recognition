import keras
from keras.layers import *
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from cnn_model.models import *
from utils.data import *
import calendar
import time

def main():
    epochs = 50
    batchSize = 15
    VALTrainingFactor = 0.7
    learningRate=0.1
    nFeaturePoints = 0 # todo
    
    files='data/labeled/**'

    cnnModel = basicCNNModel((256, 256, 3), 56)
    cnnModel.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=learningRate), metrics=['accuracy'])


    mplModel = mplModel(nFeaturePoints, 56)
    mplModel.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=learningRate), metrics=['accuracy'])


    println('training model on labeled data ')

    tbCallBack = keras.callbacks.TensorBoard(log_dir='./storage/tensor_board/labeled_training_tb_'+str(learningRate)+'_'+str(calendar.timegm(time.gmtime())), histogram_freq=0, write_graph=True, write_images=True)

    data_gen = generate_data_batches(files, batchSize, VALTrainingFactor)

    val_data_gen = generate_val_data_batches(files, batchSize, VALTrainingFactor)

    train_batch_count, val_batch_count = get_data_metric(files, batchSize, VALTrainingFactor)

    println('train_batch_count, val_batch_count: ', train_batch_count,', ', val_batch_count)

    model.fit_generator(data_gen, validation_data=val_data_gen, shuffle=True, validation_steps=val_batch_count, steps_per_epoch=train_batch_count, epochs=epochs, verbose=1, callbacks=[tbCallBack])
    model.save_weights('storage/train_labeled_weights.h5')


if __name__ == "__main__":
    main()
