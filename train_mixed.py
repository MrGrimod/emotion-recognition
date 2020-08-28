import keras
import random
from keras.layers import *
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from cnn_model.models import *
from utils.data import *
import calendar
import time

def main():
    epochs = 10
    batchSize = 16
    VALTrainingFactor = 0.7
    learningRate=0.001
    dataSetDir = 'data/MPI_simplified/**'
    files='data/labeled_MPI_simplified/**'

    classes = getClassesForDataSet(dataSetDir)

    cnnIn, cnnOutLayer = VGG16(input_shape=(48, 48, 1), nOutPut=len(classes))
    mplIn, mplOutLayer = mplModel((68, 2), len(classes))

    midModel = multipleInputDataModel(mplOutLayer, cnnOutLayer, mplIn, cnnIn, len(classes))
    midModel.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learningRate), metrics=['accuracy'])

    print('training model with mixed input on labeled data ')
    randomId = str(random.randrange(500))
    print('Model Id: ' + randomId)

    tbCallBack = keras.callbacks.TensorBoard(log_dir='data/tensorBoard/model_'+randomId, histogram_freq=0, write_graph=True, write_images=True)

    data_gen = generateMixedInputDataBatches(files, batchSize, VALTrainingFactor)

    val_data_gen = generateMixedInputValDataBatches(files, batchSize, VALTrainingFactor)

    train_batch_count, val_batch_count = getDataMetric(files, batchSize, VALTrainingFactor)

    print('train_batch_count, val_batch_count: ', train_batch_count,', ', val_batch_count)

    midModel.fit(data_gen, validation_data=val_data_gen, validation_steps=val_batch_count, steps_per_epoch=train_batch_count, epochs=epochs, verbose=1, callbacks=[tbCallBack])
    midModel.save('data/trainedModels/model_'+randomId+'.h5')

    ImageX, ImageMarkerX, batchY = getPredictionTestSample(batchSize)
    prediction = midModel.predict([ImageMarkerX, ImageX])

    for i in range(len(prediction[0])):
        print(classes[i] + ': ' + str(prediction[0][i]))
    print("-------highest propability--------")
    print(str(classes[np.argmax(prediction[0])]))
    print("-------sample class--------")
    print(classes[np.argmax(batchY)])

if __name__ == "__main__":
    main()
