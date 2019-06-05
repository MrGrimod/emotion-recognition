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

    x = np.array(data[25][0]).reshape(-1, 48, 48, 1)

    prediction = model.predict(x)

    print('Angry: ', prediction[0][0], ', Disgust: ', prediction[0][1], ', Fear:', prediction[0][2], ', Happy:', prediction[0][3], ', Sad: ', prediction[0][4], ', Surprise: ', prediction[0][5], ',  Neutral: ', prediction[0][6])

if __name__ == "__main__":
    main()
