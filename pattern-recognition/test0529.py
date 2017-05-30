from readFile import *
from keras.applications.inception_v3 import InceptionV3
from model import image_caption_model
from keras.models import load_model
from keras.optimizers import RMSprop, Adadelta, SGD
from keras.utils.layer_utils import print_summary
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from utils import *
import scipy.misc
import numpy as np
import pandas as pd
import pickle as cPickle


max_len = 10
def predict(num_of_pic,char_to_int,int_to_char,test_set):
    model = image_caption_model()
    model.load_weights('demo.h5v1.0.0_60_15_1496111479.97.h5')
    sentence = []
    imgs = []
    cur = char_to_int['$']
    img = test_set[num_of_pic]
    imgs.append(img)
    img_input = np.array(imgs)
    lang_input = cur
    #print img_input.shape
    #print np.array(lang_input).reshape(-1,1).shape
    for i in range(max_len):
        X = [img_input, np.array(lang_input).reshape(-1,1)]
        prediction = model.predict(X)
        index = np.argmax(prediction)
        if (index == char_to_int['#']):
            break
        lang_input = index
        sentence.append(int_to_char[index])
    return sentence


def main():

    (char_to_int,int_to_char)=readFile()
    valid_set = loadFile(mode = 'valid')
    for i in range(50):
        sentence = predict(i,char_to_int,int_to_char,valid_set)
        for wd in sentence:
            print wd,
        print('\n')


    '''
    print(len(int_to_char))
    print ('#:',char_to_int['#'])
    print ('$:',char_to_int['$'])
    '''

    #print train_set.shape

    #model = image_caption_model()
    #model.load_weights('demo.h5v1.0.0_57_15_1496109592.89.h5')


    #processFile(char_to_int,path="train.txt")


if  __name__ == '__main__':
    main()
