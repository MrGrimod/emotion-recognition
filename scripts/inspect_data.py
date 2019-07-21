import sys
import glob
from tqdm import tqdm
sys.path.append("..")
from utils.data import *

def main():
    i = 0
    for filename in glob.iglob('F:/emotions_detection/raw/**'):

        i += 1

        print(filename)

        data = np.load(filename)

        data_x = np.array(data[0][0])

        data_y = np.array(data[1])

        print(data_x.shape)

        for i in tqdm(range(len(data_x))):
            img = data_x[i]
            cv2.imshow('IMG', img)
            cv2.waitKey(0)

if __name__ == "__main__":
    main()
