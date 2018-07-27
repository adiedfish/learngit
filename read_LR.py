#-*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function


import numpy as np
import cPickle as pkl
import sys
import csv

loadpath = "../gcn"
features_save_filename = "features_martix"
features_save_filename_without_port = "features_martix_without_port"
n_features_save_filename = "n_features_martix"
n_features_save_filename_without_port = "n_features_martix_without_port"

labels_save_filename = "labers"
labels_for_test_save_filename = "labels_for_test". #实际上是用来训练的，名字取错懒得改了，关联太多
ipdic_save_filename = "ip_dic"

train_data_filename = "train_data"
train_val_filename = "trian_val"
test_data_filename = "test_data"
test_val_filename = "test_val"

def load_and_save_data(filename,ip_num,features_num):
	train_data = []
	test_data = []
	trian_val = []
	test_val = []
	with open(filename,'r') as f:
		features = pkl.load(f)
	print("features load done...")

	with open(loadpath+labels_save_filename,'r') as f:
		labels = pkl.load(f)
	print("labels load done...")

	n_labels = np.zeros(ip_num)
	for i in xrange(ip_num):
		if labels[i][1] == 1:
			n_labels[i] = 1

	with open(loadpath+labels_for_test_save_filename, 'r') as f:
		labels_for_train = pkl.load(f)
	print("labels for trian load done...")

	n_labels_for_train = np.zeros(ip_num)
	for i in xrange(ip_num):
		if labels_for_ttrain[i][1] == 1:
			n_labels_for_train = 1

	
	for i in xrange(ip_num):
		if n_labels_for_train[i] == 0:
			test_data.append(features[i])
			test_val.append(n_labels[i])
		else:
			train_data.append(features[i])
			trian_val.append(n_labels[i])
		if i%100000 == 0:
			sys.stdout.write("%d_________%d"%(i,ip_num))
			sys.stdout.write("\r")
			sys.stdout.flush()	
	train_data = np.array(train_data)
	test_data = np.array(test_data)
	train_val = np.array(trian_val)
	test_val = np.array(test_val)

	with open(train_data_filename,'w+') as f:
		pkl.dump(train_data,f)
	print("train data save...")
	with open(test_data_filename,'w+') as f:
		pkl.dump(test_data,f)
	print("test data save...")
	with open(train_val_filename,'w+') as f:
		pkl.dump(train_val,f)
	print("train val save...")
	with open(test_val_filename,'w+') as f:
		pkl.dump(test_val,f)
	print("test val save...")


with open("ip_num",'r') as f:
	ip_num = pkl.load(f)
load_and_save_data(loadpath+n_features_save_filename,ip_num,50)

























