from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf 
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import networkx as nx

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
sparse_save_filename = "sparse_martix"
ipset_save_filename = "ip_set"
ipdic_save_filename = "ip_dic"
features_save_filename = "features_martix"
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
#sparse_martix = sparse

support = tf.sparse_placeholder(tf.float32)
x = tf.placeholder(tf.float32)
labels = tf.placeholder(tf.float32)

b1_shape = (16)
w1_shape = (features.shape[1],16)
init_range = np.sqrt(6.0/(w1_shape[0]+w1_shape[1]))

w1 = tf.Variable(tf.random_uniform(w1_shape, minval=-init_range, maxval=init_range, dtype=tf.float32))

b1 = tf.Variable(tf.zeros(b1_shape,dtype=tf.float32))

z1 = tf.sparse_tensor_dense_matmul(support,tf.matmul(x, w1))

activate = tf.nn.relu(z1+b1)

b2_shape = (3)
w2_shape = (16,3)
init_range = np.sqrt(6.0/(w2_shape[0]+w2_shape[1]))

w2 = tf.Variable(tf.random_uniform(w2_shape, minval=-init_range, maxval=init_range, dtype=tf.float32))

b2 = tf.Variable(tf.zeros(b2_shape,dtype=tf.float32))

z2 = tf.sparse_tensor_dense_matmul(support,tf.matmul(z1, w2))

predict = tf.nn.softmax(z2+b2)


learning_rate = 0.0001
loss = sum(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=labels))

train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
epochs = 50
for i in range(epochs):
	t = time.time()
	sess.run(train_step,feed_dict={support:sparse_martix,x:features,labels:labels_for_test})
	train_loss = sess.run(loss, feed_dict={support:sparse_martix,x:features,labels:labels_for_test})
	train_acc_tf = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predict,1),tf.argmax(labels_for_test,1)),"float"))
	train_acc = sess.run(train_acc_tf,feed_dict={support:sparse_martix,x:features,labels:labels_for_test})
	
	test_loss = sess.run(loss, feed_dict={support:sparse_martix,x:features,labels:labels_all})
	test_acc_tf =tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predict,1),tf.argmax(labels_for_test,1)),"float"))
	test_acc = sess.run(test_acc_tf,feed_dict={support:sparse_martix,x:features,labels:labels_all})
	
	print("Epoch:",'%04d'%(i+1)," train_loss=","{}".format(train_loss),
		"train_acc=","{}".format(train_acc),"test_loss=","{}".format(test_loss),
		"test_acc","{}".format(test_acc),"time=","{}".format(time.time()-t))

print("Optimization Finished")




