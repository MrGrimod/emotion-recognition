import sys
from tqdm import tqdm
sys.path.append("..")
from utils.data import *

def main():
    x_train, y_train, x_test, y_test, input_shape = load_data('../dataset/ck_dataset_labeld.pickle', False)
    x_train = normalize_conv_utf8(x_train)
    for i in tqdm(range(len(x_train))):
        img = x_train[i]
        img = img[:,:,3]
        cv2.imshow('IMG', x_train[i])
        cv2.waitKey(0)

if __name__ == "__main__":
    main()
