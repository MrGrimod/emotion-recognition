import keras
import dlib
from keras.models import load_model
from imutils import face_utils
from keras.layers import *
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from cnn_model.models import *
from utils.data import *


def main():
    dataSetDir = 'data/MPI_simplified/**'

    # !important! the order of the words (their index) decides about their label number! * do not change, if not necessary *
    classes = getClassesForDataSet(dataSetDir)

    data = np.load("data/labeled_MPI_selected/19.npy", allow_pickle=True)

    img = np.array(data[0][0][20])
    dataImageMarkerX = np.array(data[1][20])

    model = load_model('data/trainedModels/model_15.h5')

    ImageX, ImageMarkerX, batchY = getPredictionTestSample(1)
    prediction = model.predict([ImageMarkerX, ImageX])

    for i in range(len(prediction[0])):
        print(classes[i] + ': ' + str(prediction[0][i]))
    print("-------highest propability--------")
    print(str(classes[np.argmax(prediction[0])]))
    

if __name__ == "__main__":
    main()
