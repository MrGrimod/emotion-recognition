# Emotion Recognition

The goal is to train a neural network with mixed input data(facial features & the img) to classify expressed emotions. The facial features are represented by single point array which mark the mout, face, .... positions which are responsible respectively the best indicator for emotions.

Raw:

![result image](media/raw.jpg)

Labeled:

![labeled image](media/labeled.jpg)

### Results

![result image](media/result.jpg)

...

## Setup

- Download the dataset from [here](https://www.b-tu.de/en/graphic-systems/databases/the-large-mpi-facial-expression-database). (You don't need to download all sets one is sufficient)

- Create the raw dataset by running
      python create_dataset.py

- Run label_data.py to create the labeled dataset
      python label_data.py

- Train the data on raw and trained and compare the results
      python train_labeled/raw.py


## Requirements

- numpy
- dlib
- pickle
- cv2
- tqdm
- imutils
- keras
- tensorflow
- collections
- sklearn

## Face recognition/ detection

The recognition and landmark detection for data labeling is done by [DLib](http://dlib.net/)

    detector = dlib.get_frontal_face_detector()


## Dataset

The requirements for the dataset are rather special because the recordings need to hace a certain resolution and quality. That's the reason why most common datasets are out of the question. The [MPI Facial Expression Database](https://www.b-tu.de/en/graphic-systems/databases/the-large-mpi-facial-expression-database) consists of 64 classes.
