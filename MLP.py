#-*- coding:utf-8 -*-
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import numpy as np 
import cPickle as pkl
import sys

train_data_filename = "../gcn/n_features_martix_without_port"
train_val_filename = "../gcn/labels_for_test"
test_data_filename = "../gcn/n_features_martix_without_port"
test_val_filename = "../gcn/labers"

with open(train_data_filename, 'r') as f:
	train_data = pkl.load(f)
print("train data load done...")
with open(train_val_filename, 'r') as f:
	train_val = pkl.load(f)
print("train val load done...")
with open(test_data_filename, 'r') as f:
	test_data = pkl.load(f)
print("test data load done...")
with open(test_val_filename, 'r') as f:
	test_val = pkl.load(f)
print("test val load done...")

seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

x = tf.placeholder(tf.float32)
labels = tf.placeholder(tf.float32)

hidden_num = 16
w1_shape = (train_data.shape[1],hidden_num)
b1_shape = (hidden_num)
init_range = np.sqrt(6.0/(w1_shape[0]+w1_shape[1]))

w1 = tf.Variable(tf.random_uniform(w1_shape,minval=-init_range,maxval=init_range,dtype=tf.float32))
b1 = tf.Variable(tf.zeros(b1_shape,dtype=tf.float32))
z1 = tf.matmul(x,w1)
activate = tf.nn.relu(z1+b1)

w2_shape = (hidden_num,2)
b1_shape = (2)
init_range = np.sqrt(6.0/(w1_shape[0]+w1_shape[1]))

w2 = tf.Variable(tf.random_uniform(w1_shape,minval=-init_range,maxval=init_range,dtype=tf.float32))
b2 = tf.Variable(tf.zeros(b1_shape,dtype=tf.float32))
z2 = tf.matmul(activate,w2)
predict = tf.nn.softmax(z2+b2)

learning_rate = 0.001
factor = 0.001

loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits:predict,labels:labels))/len(train_data)
loss += tf.reduce_sum(predict[:,1])/len(train_data)*factor

train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.initializer_all_variable()
sess = tf.Session()
sess.run(init)

epochs = 500
max_f1 = 0.0

with open("test_num",'r') as f:
	test_num = pkl.load(f)

for i in range(epochs):
	sess.run(train_step,feed_dic={x:train_data,labels:train_val})

	allb = 0
	cout = 0
	v = sess.run(predict,feed_dic={x:test_data})
	ind_all = sess.run(tf.argmax(v,1))
	for i in xrange(len(test_val)):
		if test_val[i][1] == 1:
			allb += 1
			if ind_all[i] == 1:
				cout += 1
	print("how much we predict right: %d/  %d"%(cout-test_num[1], allb-test_num[1]))
	if cout - test_num[1] >0:
		rec = (float(cout)-float(test_num[1]))/(float(allb)-float(test_num[1]))
	else:
		rec = 0
	print("blacklist predict pro:%.4f"%(rec))

	allb = 0
	cout = 0
	for i in xrange(len(test_val)):
		if ind_all[i] == 1:
			allb += 1
			if test_val[i][1] == 1:
				cout += 1
	print("how much we predict: %d/  %d"%(allb, cout-test_num[1]))
	if cout - test_num[1] >0:
		acc = (float(cout)-float(test_num[1]))/(float(allb)+1)
	else:
		acc = 0
	print("blacklist predict pro:%.4f"%(acc))
	
	if rec+acc != 0:
		f1_soc = 2*(rec*acc)/(rec+acc)
	else:
		f1_soc = 0
	if f1_soc > max_f1:
		max_f1 = f1_soc
	print(f1_soc)
	print("-------------------------------------")
























