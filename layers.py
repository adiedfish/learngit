#-*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import numpy as np 
import cPickle
import scipy.sparse as sp 
import sys

class layer(object):

	def __init__(self,name):
		self.name = name
		self.outputs = None
		self.inputs = None

class gcn_layer(layer):

	def __init__(self,name,size,support,activate_fun_num):
		super(gcn_layer,self).__init__(name)
		
		b_shape = size[0]
		w_shape = size
		init_range = np.sqrt(6.0/(w_shape[0]+w_shape[1]))
		self.w = tf.Variable(tf.random_uniform(w_shape, minval=-init_range, maxval=init_range, dtype=tf.float32))
		self.b = tf.Variable(tf.zeros(b_shape,dtype=tf.float32))
		self.support = tf.sparse_placeholder(tf.float32)
		self.activate_fun_num = activate_fun_num

	def z(activate):
		return tf.sparse_tensor_dense_matmul(self.support,tf.matmul(activate, self.w))

	def activate(z):
		if self.activate_fun_num == 0:
			return tf.nn.relu(z+self.b)
		elif self.activate_fun_num == 1:
			return tf.nn.softmax(z+self.b)

