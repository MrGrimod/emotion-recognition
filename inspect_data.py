import sys
import glob
from tqdm import tqdm
sys.path.append("..")
from utils.data import *

def main():
    i = 0
    dataSetDir = 'data/MPI_large_centralcam_hi_islf_complete/**'
    classes = getClassesForDataSet(dataSetDir)

    for filename in glob.iglob('data/labeled/**'):
        i += 1

        data = np.load(filename, allow_pickle=True)

        dataX = np.array(data[0][0])
        featurePoints = np.array(data[1])

        dataY = np.array(data[2])

        for i in tqdm(range(len(dataX))):
            img = dataX[i]

            print(str(classes[np.argmax(dataY[i])]) + ',' + str(np.argmax(dataY[i])))

            cv2.putText(img,str(classes[np.argmax(dataY[i])]), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

            for (x, y) in featurePoints[i]:
                print(x)
                print(y)
                print("------")
                # cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

            # print(featurePoints[i])

            cv2.imshow('img', img)
            cv2.waitKey(0)

if __name__ == "__main__":
    main()
