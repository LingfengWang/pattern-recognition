import os, sys
import pickle as cPickle
#import urllib.request
import h5py
import time
import pandas as pd
import scipy.misc
import numpy as np

from readFile import *
from keras.models import load_model
from utils import *
from model import image_caption_model
#from joblib import Parallel, delayed

from keras.utils.layer_utils import print_summary


'''
def gen_batch_in_thread(img_map, df_cap, vocab_size, n_jobs=4,
        size_per_thread=32):
    imgs , curs, nxts, seqs, vhists = [], [], [], [], []
    returns = Parallel(n_jobs=4, backend='threading')(
                            delayed(generate_batch)
			    (img_train, df_cap, vocab_size, size=size_per_thread) for i in range(0, n_jobs))

    for triple in returns:
        imgs.extend(triple[0])
        curs.extend(triple[1])
        nxts.extend(triple[2])
        seqs.extend(triple[3])
        vhists.extend(triple[4])

    return np.array(imgs), np.array(curs).reshape((-1,1)), np.array(nxts), \
            np.array(seqs), np.array(vhists)
'''
#text_path = 'text.h5'
def generate_batch( text_path, vocab_size=11573, size=512, max_caplen=33):
    imgs, curs, nxts = [], [], []
    train_set = loadFile()
    f = h5py.File(text_path, "r")
    text_set = f['text_set'].value

    for idx in np.random.randint(text_set.shape[0], size=size):
        row = text_set[idx]
        num_of_pic = row[0]
        for i in range(2, len(row)):
            #cur = 0
            #if (i != 1):
            cur = row[i-1]
            nxt = np.zeros((vocab_size))
            nxt[row[i]] = 1
            img = train_set[num_of_pic]

            curs.append(cur)
            nxts.append(nxt)
            imgs.append(img)
            if (row[i] == 1795):
                break
            print np.array(imgs).shape
            print np.array(curs).reshape((-1,1)).shape
    return np.array(imgs), np.array(curs).reshape((-1,1)), np.array(nxts)
if __name__ == '__main__':
    '''
    # initialization
    hist_path = 'history/'
    mdl_path = 'weights/'

    # read pkl
    dec_map = cPickle.load(open('dataset/text/dec_map.pkl', 'rb'))
    enc_map = cPickle.load(open('dataset/text/enc_map.pkl', 'rb'))
    img_train = cPickle.load(open('dataset/train_img2048.pkl', 'rb'))
    img_test = cPickle.load(open('dataset/test_img256.pkl', 'rb'))
    df_cap = pd.read_csv('dataset/text/train_enc_cap.csv')
    '''
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    vocab_size=11573
    #embedding_matrix = generate_embedding_matrix('pre_trained/glove.6B.100d.txt', dec_map)
    model = image_caption_model()

    if len(sys.argv) >= 2:
        print('load weights from : {}'.format(sys.argv[1]))
        model.load_weights(sys.argv[1])

    # insert ur version name here
    version = 'v1.0.0'
    batch_num = 30
    #print_summary(model.layers)

    hist_loss = []

    for i in range(0, 40):
        for j in range(0, batch_num):
            s = time.time()
	    # 64 x 128 = 8192 images per batch.
	    # 8 x 32 = 256 images for validation.
            img1, cur1, nxt1 = generate_batch('text.h5', vocab_size, size=4096)
            #img2, cur2, nxt2 = generate_batch(img_train, df_cap, vocab_size, size=50)
            #hist = model.fit([img1, cur1], nxt1, batch_size=32, nb_epoch=200, verbose=0, validation_data=([img2, cur2], nxt2), shuffle=True)
            hist = model.fit([img1, cur1], nxt1, batch_size=1024, nb_epoch=5, verbose=0, shuffle=True)

            # dump training history, model to disk
            mdl_path = 'demo.h5'
            #cPickle.dump({'loss':hist.history['loss'], 'val_loss':hist.history['val_loss']}, open(hist_path, 'wb'))
            model.save(mdl_path)


            print("epoch {0}, batch {1} - training loss : {2}"
                    .format(i, j, hist.history['loss'][-1]))
	    # record the training history
            #hist_loss.extend(hist.history['loss']

            if j % int(batch_num / 2) == 0 :
                print('check point')
                m_name = "{0}{1}_{2}_{3}_{4}.h5".format(mdl_path, version, i, j, time.time())
                model.save_weights(m_name)
                #cPickle.dump({'loss':hist_loss}, open(hist_path+ 'history.pkl', 'wb'))
