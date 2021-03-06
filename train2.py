#-*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import time
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
features_save_filename = "n_features_martix"
labels_save_filename = "labers"
labels_for_test_save_filename = "labels_for_test"

with open(features_save_filename,'r') as f:
	features = pkl.load(f)
	print("features load done")

with open(labels_save_filename,'r') as f:
	labels_all = pkl.load(f)
	print("labels load done")

with open(labels_for_test_save_filename,'r') as f:
	labels_for_test = pkl.load(f)
	print("labels for test load done")

with open(sparse_save_filename,'r') as f:
	sparse = pkl.load(f)
	print("sparse load done")

sparse_martix = preprocess_adj(sparse)
#sparse_martix = sparse_martix.dot(sparse_martix)
#sparse_martix = sparse

with open("ip_num",'r') as f:
	ip_num = pkl.load(f)
labels_only_background = np.zeros((ip_num,3))
background_cout = 0
labels_only_blacklist = np.zeros((ip_num,3))
blacklist_cout = 0
for i in xrange(len(labels_for_test)):
	if labels_for_test[i][0] == 1:
		labels_only_background[i][0] = 1
		background_cout += 1
	elif labels_for_test[i][1] == 1:
		labels_only_blacklist[i][1] =1
		blacklist_cout += 1

support = tf.sparse_placeholder(tf.float32)
x = tf.placeholder(tf.float32)
labels_background = tf.placeholder(tf.float32)
labels_blacklist = tf.placeholder(tf.float32)

hidden_num = 16
b1_shape = (hidden_num)
w1_shape = (features.shape[1],hidden_num)
init_range = np.sqrt(6.0/(w1_shape[0]+w1_shape[1]))

w1 = tf.Variable(tf.random_uniform(w1_shape, minval=-init_range, maxval=init_range, dtype=tf.float32))

b1 = tf.Variable(tf.zeros(b1_shape,dtype=tf.float32))

z1 = tf.sparse_tensor_dense_matmul(support,tf.matmul(x, w1))

activate = tf.nn.relu(z1+b1)

b2_shape = (3)
w2_shape = (hidden_num,3)
init_range = np.sqrt(6.0/(w2_shape[0]+w2_shape[1]))

w2 = tf.Variable(tf.random_uniform(w2_shape, minval=-init_range, maxval=init_range, dtype=tf.float32))

b2 = tf.Variable(tf.zeros(b2_shape,dtype=tf.float32))

z2 = tf.sparse_tensor_dense_matmul(support,tf.matmul(activate, w2))

predict = tf.nn.softmax(z2+b2)


learning_rate = 0.001
learning_rate_back = 0.001
lmbda = 5.0

loss_in_background = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=labels_background))/background_cout#+lmbda*(tf.reduce_sum(tf.abs(w1)))/background_cout
loss_in_blacklist = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=labels_blacklist))/blacklist_cout#+lmbda*(tf.reduce_sum(tf.abs(w2)))/blacklist_cout

factor = 0.002
loss_in_background += tf.reduce_sum(predict[:,1])/len(features)*factor
loss_in_blacklist  += tf.reduce_sum(predict[:,1])/len(features)*factor

train_step_background = tf.train.AdamOptimizer(learning_rate = learning_rate_back).minimize(loss_in_background)
train_step_blacklist = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_in_blacklist)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
epochs = 100
max_f1 = 0.0
for i in range(epochs):
	#t = time.time()
	sess.run(train_step_background,feed_dict={support:sparse_martix,x:features,labels_background:labels_only_background})
	allb = 0
	cout = 0
	v = sess.run(predict,feed_dict={support:sparse_martix,x:features})
	ind_all = sess.run(tf.argmax(v,1))
	for i in xrange(len(labels_all)):
		if labels_all[i][1] == 1:
			allb += 1
			if ind_all[i] == 1:
				cout += 1
	print("how much we predict right: %d/  %d"%(cout,allb))
	rec = float(cout)/float(allb)
	print("rec:%.4f"%(rec))

	allb = 0
	cout = 0
	for i in xrange(len(labels_all)):
		#ind = sess.run(tf.argmax(v[i]))
		if ind_all[i] == 1:
			allb += 1
			if labels_all[i][1] == 1:
				cout += 1
	print("how much we predict: %d/  %d"%(allb,cout))
	acc = float(cout)/(float(allb)+1)
	print("acc:%.4f"%(acc))

	if rec+acc != 0:
		f1_soc = 2*(rec*acc)/(rec+acc)
	els :
		f1_soc = 0
	if f1_soc>max_f1:
		max_f1 = f1_soc
	print(f1_soc)
	print("-------------------------------------")
for i in range(epochs):
	sess.run(train_step_blacklist,feed_dict={support:sparse_martix,x:features,labels_blacklist:labels_only_blacklist})
	'''
	train_loss = sess.run(loss, feed_dict={support:sparse_martix,x:features,labels:labels_for_test})
	train_acc_tf = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predict,1),tf.argmax(labels_for_test,1)),"float"))
	train_acc = sess.run(train_acc_tf,feed_dict={support:sparse_martix,x:features,labels:labels_for_test})
	
	
	test_loss = sess.run(loss, feed_dict={support:sparse_martix,x:features,labels:labels_all})
	test_acc_tf = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predict,1),tf.argmax(labels_for_test,1)),"float"))
	test_acc = sess.run(test_acc_tf,feed_dict={support:sparse_martix,x:features,labels:labels_all})
	
	
	print("Epoch:",'%04d'%(i+1)," train_loss=","{}".format(train_loss),
		"train_acc=","{}".format(train_acc),"test_loss=","{}".format(test_loss),
		"test_acc=","{}".format(test_acc),"time=","{}".format(time.time()-t))
	'''
	#print("Epoch: %04d"%(i+1))
	allb = 0
	cout = 0
	v = sess.run(predict,feed_dict={support:sparse_martix,x:features})
	ind_all = sess.run(tf.argmax(v,1))
	for i in xrange(len(labels_all)):
		if labels_all[i][1] == 1:
			allb += 1
			#v = sees.run(predict,feed_dict={support:sparse_martix,x:features})[i]
			#ind = sess.run(tf.argmax(v[i]))
			if ind_all[i] == 1:
				cout += 1
	print("how much we predict right: %d/  %d"%(cout,allb))
	rec = float(cout)/float(allb)
	print("rec:%.4f"%(rec))

	allb = 0
	cout = 0
	for i in xrange(len(labels_all)):
		#ind = sess.run(tf.argmax(v[i]))
		if ind_all[i] == 1:
			allb += 1
			if labels_all[i][1] == 1:
				cout += 1
	print("how much we predict: %d/  %d"%(allb,cout))
	acc = float(cout)/(float(allb)+1)
	print("acc:%.4f"%(acc))

	if rec+acc != 0:
		f1_soc = 2*(rec*acc)/(rec+acc)
	else:
		f1_soc = 0
	if f1_soc>max_f1:
		max_f1 = f1_soc
	print(f1_soc)
	print("-------------------------------------")

'''
allb = 0
cout = 0
v = sess.run(predict,feed_dict={support:sparse_martix,x:features})
ind_all = sess.run(tf.argmax(v,1))
for i in xrange(len(labels_all)):
	if labels_all[i][1] == 1:
		allb += 1
		#v = sees.run(predict,feed_dict={support:sparse_martix,x:features})[i]
		#ind = sess.run(tf.argmax(v[i]))
		if ind_all[i] == 1:
			cout += 1
	if allb%10 == 0:
		sys.stdout.write("%d labels done"%allb)
		sys.stdout.write('\r')
		sys.stdout.flush()
sys.stdout.write("%d labels done"%allb)
sys.stdout.flush()
rec = float(cout)/float(allb)
print("blacklist predict pro:%.4f"%(rec))

allb = 0
cout = 0
for i in xrange(len(labels_all)):
	#ind = sess.run(tf.argmax(v[i]))
	if ind_all[i] == 1:
		allb += 1
		if labels_all[i][1] == 1:
			cout += 1
	if allb%10 == 0:
		sys.stdout.write("%d labels done"%allb)
		sys.stdout.write('\r')
		sys.stdout.flush()
sys.stdout.write("%d labels done"%allb)
sys.stdout.flush()
acc = float(cout)/float(allb)
print("blacklist predict pro:%.4f"%(acc))

f1_soc = 2*(rec*acc)/(rec+acc)
print(f1_soc)
'''
print("w1 :  ---\n")
print(sess.run(w1))
print("------------------------------------")
print("w2 :  ---\n")
print(sess.run(w2))
print("Optimization Finished")
print("max f1:%.4f"%max_f1)





































