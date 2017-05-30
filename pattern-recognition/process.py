# encoding=utf-8

train_txt = open("valid_ori.txt", "r")
train_txt_marked = open("valid_marked.txt", "w")

line = train_txt.readline()
line
while line:
	if (len(line)<10):
		train_txt_marked.write(line)
		line = train_txt.readline()
		continue
	line2 = '$'+line[0:-2]+"#"+'\r\n'
	train_txt_marked.write(line2)
	line = train_txt.readline()
train_txt.close()
train_txt_marked.close()




'''
valid_txt = open("valid.txt", "r")
valid_num = []
cnt = 1;
numOfStc = 0
valid_word = set()
line = valid_txt.readline()
while line:
	if (len(line)<10):
		valid_num.append(numOfStc)
		numOfStc = 0
		cnt = cnt+1
		line = valid_txt.readline()
		continue
	numOfStc = numOfStc + 1;
	seg_list = jieba.cut(line)
	seg = " ".join(seg_list)

	for wd in seg.split():
		print wd
		valid_word.add(wd)
	line = valid_txt.readline()
valid_num.append(numOfStc)
valid_num.pop(0)

print len(word)
union_word = word | valid_word

print len(valid_num)
print len(valid_word)
print len(union_word)
print type(valid_num)
'''
'''

import h5py
import numpy as np
f = h5py.File("image_vgg19_fc2_feature_677004464.h5", "r")
test_set = f['test_set'].value
train_set = f['train_set'].value
print train_set.shape
validation_set = f['validation_set'].value
x_train = train_set.repeat(train_num,axis = 0)
print x_train.shape

max_features = 20000
maxlen = 20  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
'''
