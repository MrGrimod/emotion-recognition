import sys
import glob
from tqdm import tqdm
sys.path.append("..")
from utils.data import *

def main():
    i = 0
    for filename in glob.iglob('../data/labeled/**'):
        i += 1

        data = np.load(filename, allow_pickle=True)
        print(data.shape)
        dataX = np.array(data[0][0][0])

        featurePoints = np.array(data[0][1][0])

        dataY = np.array(data[1])

        for i in tqdm(range(len(dataX))):
            img = dataX[i]
            print(str(get_classes()[np.argmax(dataY[i])]) + ',' + str(np.argmax(dataY[i])))
            cv2.putText(img,str(get_classes()[np.argmax(dataY[i])]), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

            print(featurePoints)
            cv2.imshow('img', img)
            cv2.waitKey(0)

if __name__ == "__main__":
    main()
