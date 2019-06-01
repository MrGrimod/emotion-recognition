import csv
import os
import pandas as pd
import numpy as np
import cv2
from utils.data import *

def main():
    data = prep_data()

    img_raw = data[1][0]
    print(img_raw)
    img = cv2.cvtColor(img_raw, cv2.Color_BGR2GRAY)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    
if __name__ == '__main__':
    main()
