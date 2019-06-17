import sys
import glob
from tqdm import tqdm
sys.path.append("..")
from utils.data import *

def main():
    i = 0
    for filename in glob.iglob('F:/emotions_detection/labeled/**'):
        i += 1

        print(filename)

        data = np.load(filename)

        print(data.shape)
        
        data_x = np.array([i for i in data[0]])

        data_y = np.array([i for i in data[1]])

        print(data_x.shape)

        for i in tqdm(range(len(data_x))):
            cv2.imshow('IMG', data_x[i])
            cv2.waitKey(0)

if __name__ == "__main__":
    main()
