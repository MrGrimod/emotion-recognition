import keras
from keras.layers import *
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from cnn_model.models import *
from utils.data import *

def main():
    epochs = 1
    batch_size = 30
    val_training_factor = 0.7
    files='F:/emotions_detection/labeled/**'

    model = basic_cnn((256, 256, 3), 9)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    print('training model on labeld data \r')

    tbCallBack = keras.callbacks.TensorBoard(log_dir='./dataset/labeled_training_tb', histogram_freq=0, write_graph=True, write_images=True)

    data_gen = generate_data_batches(files, batch_size, val_training_factor)

    val_data_gen = generate_val_data_batches(files, batch_size, val_training_factor)

    train_batch_count, val_batch_count = get_data_metric(files, batch_size, val_training_factor)

    model.fit_generator(data_gen, validation_data=val_data_gen, validation_steps=val_batch_count, steps_per_epoch=train_batch_count, epochs=epochs, verbose=1)
    model.save_weights('storage/train_labeled_weights.h5')


def detect_features(x_data):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('storage/shape_predictor_68_face_landmarks.dat')

    x_final = []
    print('Detecting data - using DLib')
    for i in tqdm(range(len(x_data[0]))):
        img = x_data[0][i]
        rects = detector(img, 0)
        for (i, rect) in enumerate(rects):
        	shape = predictor(img, rect)
        	shape = face_utils.shape_to_np(shape)

        	(x, y, w, h) = face_utils.rect_to_bb(rect)

            #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        	for (x, y) in shape:
        		cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

        # cv2.imshow("Output", img)
        # cv2.waitKey(0)
        x_final.append(img)

    x_final = np.array(x_final)

    return x_final

def detect_features_cv_cascades(x_data):

    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier('cascades/haarcascade_mouth.xml')

    x_final = []
    print('Detecting data - using OpenCv')
    for i in tqdm(range(len(x_data))):
        faces = face_cascade.detectMultiScale(x_data[i], 1.3, 5)

        eyes = eye_cascade.detectMultiScale(x_data[i])
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(x_data[i],(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        mouth = mouth_cascade.detectMultiScale(x_data[i])
        for (ex,ey,ew,eh) in mouth:
            cv2.rectangle(x_data[i],(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        x_final.append(x_data[i])

    return x_final


def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# return a tuple of (x, y, w, h)
	return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords


if __name__ == '__main__':
    main()
