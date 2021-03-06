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
#features_save_filename = "n_features_martix"
features_save_filename = "n_features_martix_without_port"
labels_save_filename = "labers"
labels_for_test_save_filename = "labels_for_test"

j = 0
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


support = tf.sparse_placeholder(tf.float32)
x = tf.placeholder(tf.float32)
labels = tf.placeholder(tf.float32)

hidden_num = 16
b1_shape = (hidden_num)
w1_shape = (features.shape[1],hidden_num)
init_range = np.sqrt(6.0/(w1_shape[0]+w1_shape[1]))

w1 = tf.Variable(tf.random_uniform(w1_shape, minval=-init_range, maxval=init_range, dtype=tf.float32))

b1 = tf.Variable(tf.zeros(b1_shape,dtype=tf.float32))

z1 = tf.sparse_tensor_dense_matmul(support,tf.matmul(x, w1))
#z1 = tf.sparse_tensor_dense_matmul(support,z1)

'''
sup_sum = tf.sparse_reduce_sum(support,axis=1)
sub = sup_sum*tf.transpose(x)
z1 = tf.sparse_tensor_dense_matmul(support,tf.sparse_tensor_dense_matmul(support,x))-tf.transpose(sub)
z1 = tf.matmul(z1,w1)
'''
activate = tf.nn.relu(z1+b1)

hidden_class = 3
b2_shape = (hidden_class)
w2_shape = (hidden_num,hidden_class)
init_range = np.sqrt(6.0/(w2_shape[0]+w2_shape[1]))

w2 = tf.Variable(tf.random_uniform(w2_shape, minval=-init_range, maxval=init_range, dtype=tf.float32))

b2 = tf.Variable(tf.zeros(b2_shape,dtype=tf.float32))

z2 = tf.sparse_tensor_dense_matmul(support,tf.matmul(activate, w2))
#z2 = tf.sparse_tensor_dense_matmul(support,z2)
'''
sup_sum = tf.sparse_reduce_sum(support,axis=1)
sub = sup_sum*tf.transpose(activate)
z2 = tf.sparse_tensor_dense_matmul(support,tf.sparse_tensor_dense_matmul(support,activate))-tf.transpose(sub)
z2 = tf.matmul(z2,w2)
'''
predict = tf.nn.softmax(z2+b2)
'''
activate = tf.nn.relu(z2+b2)

out_class = 3
b3_shape = (out_class)
w3_shape = (hidden_class,out_class)
init_range = np.sqrt(6.0/(w2_shape[0]+w2_shape[1]))

w3 = tf.Variable(tf.random_uniform(w3_shape, minval=-init_range, maxval=init_range, dtype=tf.float32))

b3 = tf.Variable(tf.zeros(b3_shape,dtype=tf.float32))

z3 = tf.matmul(activate, w3)

predict = tf.nn.softmax(z3+b3)

'''

learning_rate = 0.005
lmbda = 5.0
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=labels))/len(features)#+lmbda*(tf.reduce_sum(tf.abs(w1))+tf.reduce_sum(tf.abs(w2)))/len(features)
#权值已经够小了不用正则项约束
factor = 0.005
loss_1 = loss + tf.reduce_sum(predict[:,1])/len(features)*factor
train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
epochs = 500
max_f1 = 0.0

with open("test_num",'r') as f:
	test_num = pkl.load(f)


for i in range(epochs):
	#t = time.time()

	sess.run(train_step,feed_dict={support:sparse_martix,x:features,labels:labels_for_test})
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
	for i in xrange(len(labels_all)):
		#ind = sess.run(tf.argmax(v[i]))
		if ind_all[i] == 1:
			allb += 1
			if labels_all[i][1] == 1:
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
	print("%.5f       %d"%(f1_soc,j))
	print("-------------------------------------")

print("w1 :  ---\n")
w_1 = sess.run(w1)
print(w_1)
print("------------------------------------")
print("w2 :  ---\n")
w_2 = sess.run(w2)
print(w_2)
print("Optimization Finished")
print("max f1:%.4f"%max_f1)

'''
with open("fin_pre/fin_w1_"+str(j),'w+') as f:
	pkl.dump(w_1,f)
b_1 = sess.run(b1)
with open("fin_pre/fin_b1_"+str(j),'w+') as f:
	pkl.dump(b_1,f)
with open("fin_pre/fin_w2_"+str(j),'w+') as f:
	pkl.dump(w_2,f)
b_2 = sess.run(b2)
with open("fin_pre/fin_b2_"+str(j),'w+') as f:
	pkl.dump(b_2,f)
	
fin_pre = sess.run(predict,feed_dict={support:sparse_martix,x:features})
fin_pre = sess.run(tf.argmax(fin_pre,1))

with open("fin_pre/fin_pre_"+str(j),'w+') as f:
	pkl.dump(fin_pre,f)
'''






