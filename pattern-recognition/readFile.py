# encoding=utf-8
import jieba
import numpy as np
import os
import sys
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation
from keras.utils import np_utils
import h5py


def readFile(path="train.txt"):
	train_txt = open(path, "r")
	train_num = []
	count = 0
	cnt = 1;
	numOfStc = 0
	max_size = 0
	batch_size = 2048
	#idx = []
	word = set()
	line = train_txt.readline()
	while line:
		if (len(line)<10):
			train_num.append(numOfStc)
			numOfStc = 0
			cnt = cnt+1
			line = train_txt.readline()
			continue
		numOfStc = numOfStc + 1;
		seg_list = jieba.cut(line)
		seg = " ".join(seg_list)

	#	count = 0
		for wd in seg.split():
			count = count + 1
	#		print wd
			word.add(wd)
	#	if (count > max_size):
	#		max_size = count
		line = train_txt.readline()
	train_num.append(numOfStc)
	train_num.pop(0)
	char_to_int = dict((c, i) for i, c in enumerate(word))
	int_to_char = dict((i, c) for i, c in enumerate(word))
	num_of_type = len(char_to_int)
	train_txt.close()
	return (char_to_int,int_to_char)

def loadFile(path="image_vgg19_fc2_feature_677004464.h5",mode = 'train'):
	f = h5py.File(path, "r")
	if (mode == 'train'):
		train_set = f['train_set'].value
		return train_set
	else:
		if (mode == 'test'):
			test_set = f['test_set'].value
			return test_set
	#print train_set.shape
		else:
			validation_set = f['validation_set'].value
			return validation_set

def processFile(char_to_int,path="train.txt"):
	train_txt = open(path, "r")
	cnt = -1;
	count = 0
	line = train_txt.readline()
	text_set = np.zeros((38445,34),dtype=int)
	while line:
		if (len(line)<10):
			cnt = cnt+1
			line = train_txt.readline()
			continue
		seg_list = jieba.cut(line)
		seg = " ".join(seg_list)
		text_set[count][0] = cnt
		ls = seg.split()
		for i in range(len(ls)):
			text_set[count][i+1] = char_to_int[ls[i]]
		count = count + 1
		line = train_txt.readline()
	train_txt.close()
	h5file = h5py.File('text.h5','w')
	h5file.create_dataset('text_set', data = text_set)


	#x_train = train_set.repeat(train_num,axis = 0)

	#print train_set.dtype



'''
#print max_size = 31
#print len(word)
#print count
#sequences = np.zeros((38445,31),dtype = float32)sequences[row][col] = char_to_int[wd]
print num_of_type
f = h5py.File("image_vgg19_fc2_feature_677004464.h5", "r")
test_set = f['test_set'].value
train_set = f['train_set'].value
#print train_set.shape
validation_set = f['validation_set'].value
#x_train = train_set.repeat(train_num,axis = 0)

#print train_set.dtype




def generate_arrays_from_file(path,batch_size,trainSet=train_set):
        raw_row = -1
#        new_row = 0
        while 1:
                f = open(path,'r')
                cnt = 0
                X =[]
                Y =[]
                y = np.zeros([1,num_of_type],int)
                x= np.zeros([1,4096+num_of_type],float)
                for line in f:
                        if (len(line)<10):
                                raw_row = raw_row + 1
                                continue
                        seg_list = jieba.cut(line)
                        seg = " ".join(seg_list)
                        last_wd = ''
                        for wd in seg.split():
                                x[:4096] = trainSet[raw_row]
                        if (last_wd!=''):
                                x[4096+char_to_int[last_wd]] = 1
                                y[char_to_int[wd]] = 1
                                last_wd = wd
                        print(x)
                X.append(x)
                Y.append(y)
                cnt += 1
                if cnt==batch_size:
                        cnt = 0
                        print(np.array(X))
                        print(np.array(Y))
                        yield(np.array(X), np.array(Y))
                        X = []
                        Y = []

        f.close()

for i in range(10):
        print(generate_arrays_from_file("train.txt",batch_size=32))
'''



'''
for i in range(10):
        gen = generate_arrays_from_file("train_txt",32))
        print type(gen)

sequences = np.zeros([count,1],int)
x_train = np.zeros([count,4097],float)

train_txt = open("train.txt", "r")
raw_row = -1
new_row = 0
line = train_txt.readline()
while line:
	if (len(line)<10):
		line = train_txt.readline()
		raw_row = raw_row + 1
		continue
	seg_list = jieba.cut(line)
	seg = " ".join(seg_list)
	last_wd = ''
	for wd in seg.split():
		x_train[new_row][:4096] = train_set[raw_row]
		if (last_wd!=''):
			x_train[new_row][4096] = char_to_int[last_wd]
		sequences[new_row] = char_to_int[wd]
		new_row = new_row +1
#		col = col + 1
#		print wd
		last_wd = wd
	line = train_txt.readline()
#	col = 0
train_txt.close()
print sequences;
print x_train;

global length = 2048
global tms_max = count/length+1
global tms = -1

def Generator(x = x_train, y = sequences):
	tmp = (tms + 1) % tms_max
	tms = tmp
	if (tms != tms_max):
		x_data = x[tms*length:(tms+1)*length]
	else:
		x_data = x[tms*length:]

	y_data =  np.zeros([x_data.shape(0),num_of_type],int)
	for i in range(x_data.shape(0)):
		y_data[i][sequences[tms*length]] = 1
	return (x_data,y_data)

for i in range(10):
	print Generator();


seq_length = 4097
x_train_tmp = np.reshape(x_train, (n_patterns, seq_length, 1))
x_train_tmp = np.reshape(None, x_train[:100], (len(x_train[:100]), 4097))
keras.preprocessing.text.one_hot(sequences, n,
    filters=base_filter(), lower=True, split=" ")
#y_train_tmp = np_utils.to_categorical(sequences)
#y_train

epochs = 10
batch_size = 64
model = Sequential()
model.add(LSTM(128,input_shape=(1,(4096+num_of_type))))
model.add(Dense(64, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(generate_arrays_from_file("train.txt",batch_size=batch_size),steps_per_epoch=count, epochs=epochs)

#model.fit_generator(generate_arrays_from_file(),batch_size=batch_size,steps_per_epoch=length, epochs=epochs,validation_data=(x_test, y_test))

#model.fit(x_train_tmp, y_train_tmp, nb_epoch=1000, batch_size=1, verbose=2)
#scores = model.evaluate(x_train_tmp, y_train_tmp, verbose=0)
#print("Model Accuracy: %.2f%%" % (scores[1]*100))

print sequences
print x_train

#38445 sentences
'''
