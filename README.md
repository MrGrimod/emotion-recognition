# Emotion Recognition

The goal is to get an idea of the improvement done by datasets in which the image data is additionally labeled with eyes, mouth, etc. data, tracked by opencv.

## Setup

- Download the dataset from [here](https://www.b-tu.de/en/graphic-systems/databases/the-large-mpi-facial-expression-database). (You don't need to download all sets)

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

The requirements for the dataset are rather special because the recordings need to hace a certain resolution and quality. That's the reason why most common datasets are out of the question. The [MPI Facial Expression Database](https://www.b-tu.de/en/graphic-systems/databases/the-large-mpi-facial-expression-database) consists of 64 classes, which can be summarized to roughly 7 classes.



- **agree**:
Agree_considered, Agree_continue, Agree_pure, Agree_reluctant

- **annoyed**:
Annoyed_bothered, Annoyed_rolling-eyes, Arrogant

- **i_did_not**:
I_did_not_hear,I_do_not_care,I_do_not_know,I_do_not_understand

- **negative**:
Not_convinced,Remember_negative,Imagine_negative, Disagree_considered,Disagree_pure,Disagree_reluctant,Disbelief,Pain_felt,Pain_seen,Sad,Confused

- **positive**: Imagine_positiv,Remember_positiv,Happy_achievement,Happy_laughing,Happy_satiated,Happy_schadenfreude,Impressed

- **fear**: Disgust,Contempt,Fear_oops,Fear_terror

- **smiling**: Smiling_sardonic,Smiling_triumphant,Smiling_uncertain,Smiling_winning,Smiling_yeah-right, Smiling_encouraging,Smiling_endearment,Smiling_flirting,Smiling_sad-nostalgia
