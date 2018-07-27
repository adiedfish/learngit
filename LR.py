#-*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function


import numpy as np
import cPickle as pkl
import sys

#牛顿法求最优解,特征最后一项加1，w里包含b

train_data_filename = "train_data"
train_val_filename = "trian_val"
test_data_filename = "test_data"
test_val_filename = "test_val"

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

w_shape = 50
w = np.random.randn(w_shape)

epochs = 100

def p_pos(x,w):
	e = np.exp(np.dot(w,x))
	return e/1+e
def p_neg(x,w):
	pass

def cal_odds(x,w):
	In_p_pos_p_neg = np.dot(w,x)
	return In_p_pos_p_neg

def predict(x,w,p):
	odds = cal_odds(x,w)
	if odds >= p:
		return 1
	else:
		return 0

for i in range(epochs):
	derivate_1 = np.zeros(w_shape)
	derivate_2 = 0
	for j in range(train_data.shape[0]):
		x = train_data[j]
		y = train_val[j]
		p = p_pos(x,w)
		derivate_1 -= x*(y-p)
		derivate_2 += np.dot(x,x)*p*(1-p)
		if j%10 == 0:
			sys.stdout.write("%d add done______(in %d)"%(j,train_data.shape[0]))
			sys.stdout.write("\r")
			sys.stdout.flush()
	w -= derivate_1/derivate_2
	if i%10 == 0:
		sys.stdout.write("%d epochs done______(in %d)"%(i,epochs))
		sys.stdout.write("\r")
		sys.stdout.flush()

p_for_test = 1
cout = 0
cout_2 = 0
right_cout = 0
for i in xrange(test_data.shape[0]):
	x = test_data[i]
	y = test_val[i]
	pre = predict(x,w,p)
	if y == 1:
		cout += 1
	if pre == 1:
		cout_2 += 1
	if pre == y and y == 1:
		right_cout += 1
print("\ntest data num: %d"%test_data.shape[0])
print("we predict: %d"%cout_2)
print("we predict right: %d(of %d)"%(right_cout,cout))













