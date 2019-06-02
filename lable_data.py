import csv
import os
import pandas as pd
import numpy as np
import cv2
from utils.data import *

def main():
    data = prep_data()

    img_raw = np.array(data[10][0])
    print(img_raw)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img_raw)
    cv2.resizeWindow('image', 100,100)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
