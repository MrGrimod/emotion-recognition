import csv
import os
import numpy as np
import cv2

def main():
    data_path = "dataset/fer2013.csv"
    with open(data_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            img = cv2.imdecode(np.array(row[1]), cv2.IMREAD_GRAYSCALE)
            print(img)
            cv2.imshow('img', img)
            cv2.waitKey(3000)

if __name__ == '__main__':
    main()
