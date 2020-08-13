import keras
from keras.layers import *
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from cnn_model.models import *
from utils.data import *
import calendar
import random
import time

def main():
    epochs = 10
    batchSize = 32
    VALTrainingFactor = 0.7
    learningRate = 0.001
    dataSetDir = 'data/MPI_selected/**'
    files='data/labeled_MPI_selected/**'

    classes = getClassesForDataSet(dataSetDir)

    cnnIn, cnnOut = VGG16(input_shape=(48, 48, 1), nOutPut=len(classes))
    model = Model(inputs=cnnIn, outputs=cnnOut)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=learningRate), metrics=['accuracy'])

    print('training model on raw unmarked data \r')
    randomId = str(random.randrange(500))
    print('Model Id: ' + randomId)

    tbCallBack = keras.callbacks.TensorBoard(log_dir='data/tensorBoard/raw_training_tb_'+str(learningRate)+'_'+randomId, histogram_freq=0, write_graph=True, write_images=True)

    data_gen = generateDataBatches(files, batchSize, VALTrainingFactor)

    val_data_gen = generateValDataBatches(files, batchSize, VALTrainingFactor)

    train_batch_count, val_batch_count = getDataMetric(files, batchSize, VALTrainingFactor)

    print('train_batch_count, val_batch_count: ', train_batch_count,', ', val_batch_count)

    model.fit(data_gen, validation_data=val_data_gen, validation_steps=val_batch_count, steps_per_epoch=train_batch_count, epochs=epochs, verbose=1, callbacks=[tbCallBack])
    
    model.save_weights('data/trainedModels/train_raw_weight_'+randomId+'.h5')

if __name__ == "__main__":
    main()
