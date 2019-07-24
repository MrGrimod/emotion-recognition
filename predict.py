import keras
from keras.layers import *
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from cnn_model.models import *
from utils.data import *


def main():
    model = VGG_16((256, 256, 3), 56)

    # !important! the order of the words (their index) decides about their label number! * do not change, if not necessary *
    classes = get_classes()
    model.load_weights('storage/train_raw_weights.h5')

    img = cv2.imread('storage/dataset/MPI_large_centralcam_hi_cawm_complete/Subset 01/agree_considered/cawm_agree_considered_001.png', cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))

    img = img.reshape(-1,256,256,3)

    print(img.shape)

    prediction = model.predict(img)

    print(prediction.shape)

    for i in range(len(prediction[0])):
        print(classes[i] + ': ' + str(prediction[0][i]))

if __name__ == "__main__":
    main()
