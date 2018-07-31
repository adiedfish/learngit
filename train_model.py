#-*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import numpy as np
import cPickle as pkl
import scipy.sparse as sp
import networkx as nx
import sys


def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

savepath = ""
save_filename = "clear_data.csv"
sparse_save_filename = "scale_sparse"
ipset_save_filename = "ip_set"
ipdic_save_filename = "ip_dic"
#features_save_filename = "n_features_martix"
features_save_filename = "n_features_martix_without_port"
labels_save_filename = "labers"
labels_for_test_save_filename = "labels_for_test"

with open(features_save_filename,'r') as f:
	features = pkl.load(f)
	print("features load done")

with open(labels_save_filename,'r') as f:
	labels_all = pkl.load(f)
	print("labels load done")

with open(labels_for_test_save_filename,'r') as f:
	labels_for_train = pkl.load(f)
	print("labels for test load done")

with open(sparse_save_filename,'r') as f:
	sparse = pkl.load(f)
	print("sparse load done")

sparse_martix = preprocess_adj(sparse)

gcn = model.Model()
hidden_num = 15
layer1 = layers.gcn_layer("1",(features.shape[1],hidden_num),0)
layer2 = layers.gcn_layer("2",(hidden_num,3),1)
gcn.build_layer((layer1,layer2))

epochs = 500

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(epochs):
	sess.run(gcn.train(0.001),feed_dict={layer1.support:sparse_martix,layer2.support:sparse_martix,gcn.inputs:features,gcn.labels_for_train:labels_for_train})
	p = sess.run(gcn.predict(),feed_dict={layer1.support:sparse_martix,layer2.support:sparse_martix,gcn.inputs:features})
	ind_all = sess.run(tf.argmax(p,1))
	allb = 0
	cout = 0
	for i in xrange(len(labels_all)):
		if labels_all[i][1] == 1:
			allb += 1
			if ind_all[i] == 1:
				cout += 1
	print("how much we predict right: %d/  %d"%(cout, allb))
	if cout>0:
		rec = float(cout)/(float(allb)
	else:
		rec = 0
	print("blacklist predict pro:%.4f"%(rec))






























