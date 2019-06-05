import keras
from keras.layers import *
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from cnn_model.models import *
from utils.data import *


def main():
    data = prep_data()

    model = basic_cnn((48, 48, 1), 7)

    model.load_weights('dataset/train_raw_weights.h5')

    model.predict()


if __name__ == "__main__":
    main()
