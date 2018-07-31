#-*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import numpy as np 
import cPickle
import scipy.sparse as sp 
import sys

class Model(object):

	def __init__(self):
		self.layers = []

		self.inputs = tf.placeholder(tf.float32)
		self.outputs = None

		self.labels_for_train = tf.placeholder(tf.float32)
		self.labels_for_test = tf.placeholder(tf.float32)

		self.loss = 0
		self.acc = 0
		self.rec = 0

		self.learning_rate = 0.001
		self.epochs = 500
		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

	def build_layers(self,layers):
		for layer in layers:
			self.layers.append(layer)
		activate = self.inputs
		for layer in self.layers:
			z = layer.z(activate)
			activate = layer.activate(z)
		self.outputs = activate

	def change_layer(self,layer,i):
		self.layers[i] = layer		

	def _build_loss(self):
		pre = self.predict()
		labels = self.labels_for_train
		self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=pre,labels=labels))
	
	def train(self,learning_rate):
		self._build_loss()
		self.learning_rate = learning_rate
		self.optimizer.minimize(self.loss)

	def predict(self):
		return self.outputs











