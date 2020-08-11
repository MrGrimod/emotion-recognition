import keras
import dlib
from imutils import face_utils
from keras.layers import *
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from cnn_model.models import *
from utils.data import *


def main():
    dataSetDir = "data/MPI_selected"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

    # !important! the order of the words (their index) decides about their label number! * do not change, if not necessary *
    classes = getClassesForDataSet(dataSetDir)


    img = cv2.imread(dataSetDir+'/Subset 06/disgust/islf_disgust_001.png', cv2.IMREAD_COLOR)
    
    img = detect_face(img, detector, predictor)
    if type(img) is np.ndarray:
        img = cv2.resize(img, (48, 48))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = detector(img, 0)
    
    for (i, rect) in enumerate(rects):
        shape = predictor(img, rect)
        imgFeaturePoints = np.array(face_utils.shape_to_np(shape), int)
        break

    cnnIn, cnnOutLayer = VGG16(input_shape=(48, 48, 1), nOutPut=len(classes))
    mplIn, mplOutLayer = mplModel((68, 2), len(classes))

    midModel = multipleInputDataModel(mplOutLayer, cnnOutLayer, mplIn, cnnIn, len(classes))

    midModel.load_weights('data/trainedModels/train_labeled_weights_49.h5')

    prediction = midModel.predict(img)

    for i in range(len(prediction[0])):
        print(classes[i] + ': ' + str(prediction[0][i]))

if __name__ == "__main__":
    main()
