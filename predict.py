import keras
from keras.layers import *
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from cnn_model.models import *
from utils.data import *


def main():
    model = VGG_16((256, 256, 3), 56)

    # !important! the order of the words (their index) decides about their label number! * do not change, if not necessary *
    classes = ['agree_considered','agree_continue','agree_pure','agree_reluctant','aha-light_bulb_moment','annoyed_bothered','annoyed_rolling-eyes','arrogant','bored','compassion','confused','contempt','I_did_not_hear','I_dont_care','I_do_not_care','I_dont_understand','I_dont_know','I_do_not_know','I_do_not_understand','disagree_considered','disagree_pure','disagree_reluctant','disbelief','disgust','embarrassment','fear_oops','fear_terror','happy_achievement','happy_laughing','happy_satiated','happy_schadenfreude','imagine_negative','imagine_positiv','impressed','insecurity','not_convinced','pain_felt','pain_seen','sad','remember_negative','smiling_sardonic','remember_positiv','thinking_considering','thinking_problemsolving','treudoof_bambi-eyes','smiling_encouraging','smiling_endearment','smiling_flirting','smiling_sad-nostalgia','smiling_triumphant','smiling_uncertain','smiling_winning','smiling_yeah-right','tired','treudoof ("bambi-eyes")', 'thinking_problem-solving'];

    model.load_weights('storage/train_raw_weights.h5')

    img = cv2.imread('storage/dataset/MPI_large_centralcam_hi_cawm_complete/Subset 01/agree_considered/cawm_agree_considered_001.png', cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))

    img = img.reshape(-1,256,256,3)

    print(img.shape)

    prediction = model.predict(img)

    print(prediction.shape)

    for i in range(len(prediction[0])):
        print(classes[i] + ': ' + str(prediction[0][i]))

if __name__ == "__main__":
    main()
