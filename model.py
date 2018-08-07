import os
import time
import logging
import random
import numpy as np
import tensorflow as tf
import math
from tensorflow.python.ops import variable_scope as vs
from matplotlib import pyplot as plt
from nets_classification import *


class DistrSystem(object):
	def __init__(self, FLAGS, labels, institutions):
		logging.basicConfig(filename=FLAGS.log, filemode='a', level=logging.INFO)
		self.FLAGS = FLAGS
		self.labels_dict = labels
		self.inst_dict = institutions

		self.images = tf.placeholder(tf.float32, shape=[None, FLAGS.img_height, FLAGS.img_width, FLAGS.img_channels])
		self.labels = tf.placeholder(tf.int32, shape=[None])
		self.loss_weights = tf.placeholder(tf.float32, shape=[None])
		self.lr = tf.placeholder(tf.float32)
		self.is_training = tf.placeholder(tf.bool)
		self.keep_prob = tf.placeholder(tf.float32)

		if FLAGS.model == 'linear':
			self.setup_linear()
		elif FLAGS.model == 'googlenet':
			self.setup_googlenet()
		elif FLAGS.model == 'resnet':
			self.setup_resnet50()
		self.setup_loss()   

		global_step = tf.Variable(0, trainable=False)
		learning_rate = tf.train.exponential_decay(self.lr, global_step, 
			round(len(institutions)/self.FLAGS.batch_size/self.FLAGS.num_inst), self.FLAGS.decay)

		self.optimizer = tf.train.AdamOptimizer(learning_rate)
		self.updates = self.optimizer.minimize(self.loss)
		self.saver = tf.train.Saver()
	
	def setup_linear(self):
		x = tf.reshape(self.images, shape=[-1, self.FLAGS.img_height*self.FLAGS.img_width*self.FLAGS.img_channels])
		W = tf.get_variable('W', shape=[self.FLAGS.img_height*self.FLAGS.img_width*self.FLAGS.img_channels, self.FLAGS.num_classes], 
			initializer=tf.contrib.layers.xavier_initializer())
		b = tf.get_variable('b', shape=[self.FLAGS.num_classes])
		self.predictions = tf.matmul(x, W) + b
	   
	def setup_googlenet(self):
		self.predictions = GoogLe_Net(self.images, self.is_training, self.FLAGS.num_classes, self.FLAGS.batch_size, keep_prob=self.keep_prob)
		
	def setup_resnet50(self):
		self.predictions = Res_Net(self.images, self.is_training, self.FLAGS.num_classes, self.FLAGS.batch_size, keep_prob=self.keep_prob)    

	def setup_loss(self):
		self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.predictions, weights=self.loss_weights)      

	def accuracy(self, logits, truth):
		predictions = np.argmax(logits, axis=1)
		return np.mean(0.0 + (predictions == truth))

	def graph(self, train_iters, train_vals, val_iters, val_vals, y_label):
		fig = plt.figure()
		plt.xlabel('Epochs')
		plt.ylabel(y_label)
		plt.plot(train_iters / (len(self.inst_dict) / (self.FLAGS.num_inst * self.FLAGS.batch_size)), train_vals)
		plt.plot(val_iters / (len(self.inst_dict) / (self.FLAGS.num_inst * self.FLAGS.batch_size)), val_vals)
		if y_label == 'Loss':      
			fig.savefig(self.FLAGS.loss_curve, bbox_inches='tight')
		else:
			fig.savefig(self.FLAGS.acc_curve, bbox_inches='tight')


	def augment(self, data_iter):
		matrix_size = data_iter.shape[0]
		roller = np.round(float(matrix_size/7))
		ox, oy = np.random.randint(-roller, roller+1, 2)
		# do_flip = np.random.randn() > 0
		# num_rot = np.random.choice(4)
		# pow_rand = np.clip(0.05*np.random.randn(), -.2, .2) + 1.0
		add_rand = np.clip(np.random.randn() * 0.1, -.4, .4)
		# Rolling
		data_iter = np.roll(np.roll(data_iter, ox, 0), oy, 1)

		# if do_flip:
		#     data_iter = np.fliplr(data_iter)

		# data_iter = np.rot90(data_iter, num_rot)

		#data_iter = data_iter ** pow_rand

		data_iter += add_rand
		return data_iter


	def optimize(self, session, sample, inst):
		y = [self.labels_dict[i] for i in sample]
		x = np.zeros([len(sample), self.FLAGS.img_height, self.FLAGS.img_width, self.FLAGS.img_channels], dtype=np.float32)
		for i in range(len(sample)):
			img = np.load(os.path.join(self.FLAGS.train, sample[i]))
			if self.FLAGS.augment:
				x[i] = self.augment(img)
			else:
				x[i] = img
		input_feed = {}
		input_feed[self.images] = x
		input_feed[self.labels] = y
		input_feed[self.lr] = self.inst_lr[inst]
		input_feed[self.loss_weights] = [self.loss_weights_dict[i] for i in sample]
		input_feed[self.keep_prob] = 1.-self.FLAGS.dropout
		input_feed[self.is_training] = True
		output_feed = [self.updates, self.loss, self.predictions]
		outputs = session.run(output_feed, input_feed)
		return outputs[1], self.accuracy(outputs[2], np.asarray(y, dtype=int))

	def test(self, session, test_set, val):
		if val:
			logging.info('VALIDATING')
		else:
			logging.info('TESTING')
		y = [self.labels_dict[i] for i in test_set]
		batch_indices = [0]
		index = 0
		while (True):
			index += 1
			if index == len(test_set):
				if batch_indices[-1] != index:
					batch_indices.append(index)
				break
			if index % self.FLAGS.batch_size == 0:
				batch_indices.append(index)
		num_minibatches = len(batch_indices) - 1
		losses = []
		accuracies = []
		for b_end in range(1, num_minibatches + 1):
			start = batch_indices[b_end-1]
			end = batch_indices[b_end]
			test_set_batch = test_set[start:end]
			y_batch = y[start:end]  
			x = np.zeros([end-start, self.FLAGS.img_height, self.FLAGS.img_width, self.FLAGS.img_channels], dtype=np.float32)
			for i in range(len(test_set_batch)):
				img = None
				if val:
					img = np.load(os.path.join(self.FLAGS.val, test_set_batch[i]))
				else:
					img = np.load(os.path.join(self.FLAGS.test, test_set_batch[i]))
				x[i] = img


			input_feed = {}

			input_feed[self.images] = x
			input_feed[self.labels] = y_batch
			input_feed[self.loss_weights] = [1.0 for i in test_set_batch]
			input_feed[self.keep_prob] = 1.
			input_feed[self.is_training] = False         

			output_feed = [self.loss, self.predictions]

			outputs = session.run(output_feed, input_feed)
			losses += [outputs[0]]
			accuracies += [self.accuracy(outputs[1], np.asarray(y_batch, dtype=int))]


		return sum(losses)/len(losses), sum(accuracies)/len(accuracies)  

	def get_loss_weights(self, train_sets):
		self.loss_weights_dict = {}
		if self.FLAGS.weighted_loss:            
			for i in train_sets:
				inst_set = list(train_sets[i])
				inst_labels = [self.labels_dict[img] for img in inst_set]
				label_num = [0 for l in range(self.FLAGS.num_classes)]
				for l in inst_labels:
					label_num[l] += 1
				label_weights = [len(inst_set)/(label_num[l]*self.FLAGS.num_classes) for l in range(self.FLAGS.num_classes)]
				for l in range(len(inst_set)):
					self.loss_weights_dict[inst_set[l]] = label_weights[self.labels_dict[inst_set[l]]]
		else:
			self.loss_weights_dict = {img: 1.0 for img in self.inst_dict}

	def get_sample_weights(self, train_sets):
		self.sample_weights_dict = {}
		for i in train_sets:
			inst_set = list(train_sets[i])
			if self.FLAGS.weighted_samples:            
				inst_labels = [self.labels_dict[img] for img in inst_set]
				label_num = [0 for l in range(self.FLAGS.num_classes)]
				for l in inst_labels:
					label_num[l] += 1
				sample_weights = [1./(label_num[l]*self.FLAGS.num_classes) for l in range(self.FLAGS.num_classes)]
				for l in range(len(inst_set)):
					self.sample_weights_dict[inst_set[l]] = sample_weights[self.labels_dict[inst_set[l]]]
			else:
				for img in inst_set:
					self.sample_weights_dict[img] = 1./len(inst_set)

	def get_num_iters_per_cycle(self, train_sets):
		self.inst_iters_per_cycle = [0 for i in range(self.FLAGS.num_inst)]
		if self.FLAGS.prop_iter:
			for i in train_sets:
				self.inst_iters_per_cycle[i] = round(len(train_sets[i])/self.FLAGS.batch_size*self.FLAGS.num_epochs)
		else:
			num_train_samples = len(self.inst_dict)
			self.inst_iters_per_cycle = [round(num_train_samples*self.FLAGS.num_epochs/(self.FLAGS.batch_size*self.FLAGS.num_inst)) for i in range(len(self.FLAGS.num_inst))]

	def get_lr(self, train_sets):
		self.inst_lr = [self.FLAGS.lr for i in range(self.FLAGS.num_inst)]
		if self.FLAGS.prop_lr:
			num_train_samples = len(self.inst_dict)
			for i in train_sets:
				self.inst_lr[i] = len(train_sets[i]) / num_train_samples * self.FLAGS.lr * self.FLAGS.num_inst

	def get_train_sample(self, train_set):
		train_list = list(train_set)
		weights = [self.sample_weights_dict[x] for x in train_set]
		sample = np.random.choice(train_list, self.FLAGS.batch_size, p=weights)
		return sample

	def train(self, session, train_sets, val_set):
		logging.info("training:")
		self.get_loss_weights(train_sets)
		self.get_sample_weights(train_sets)
		self.get_num_iters_per_cycle(train_sets)
		self.get_lr(train_sets)

		best_val_loss = np.inf
		train_iters = []
		train_losses = []
		train_accs = []
		val_iters = []
		val_losses = []
		val_accs = []

		tot_iters = 0
		for cycle in range(self.FLAGS.max_cycles):
			for inst in range(self.FLAGS.num_inst):
				inst_iters = self.inst_iters_per_cycle[inst]
				for i in range(inst_iters):
					sample = self.get_train_sample(train_sets[inst])
					train_loss, train_acc = self.optimize(session, sample, inst)
					logging.info('Iter: %s, Cycle: %s, Inst: %s, train loss: %.3f, train acc: %.3f' % (tot_iters, cycle, inst, train_loss, train_acc))
					tot_iters += 1
			if self.FLAGS.val is not None:
				if (cycle + 1) % self.FLAGS.val_freq == 0:
					train_sample = np.random.choice(self.inst_dict.keys(), self.FLAGS.val_size)
					val_sample = np.random.choice(list(val_set), self.FLAGS.val_size)
					logging.info('='*90)
					train_loss, train_acc = self.test(session, train_sample, True)
					train_iters.append(tot_iters)
					train_losses.append(train_loss)
					train_accs.append(train_acc)
					val_loss, val_acc = self.test(session, val_sample, True)
					val_iters.append(tot_iters)
					val_losses.append(val_loss)
					val_accs.append(val_acc)					
					if self.FLAGS.loss_curve is not None:
						self.graph(np.asarray(train_iters, dtype=float), train_losses, np.asarray(val_iters, dtype=float), val_losses, 'Loss')
					if self.FLAGS.acc_curve is not None:
						self.graph(np.asarray(train_iters, dtype=float), train_accs, np.asarray(val_iters, dtype=float), val_accs, 'Accuracy')
					logging.info('Cycle: %s, train loss: %.3f, train acc: %.3f' % (cycle, train_loss, train_acc))
					logging.info('Cycle: %s, val loss: %.3f, val acc: %.3f' % (cycle, val_loss, val_acc))
					if val_loss < best_val_loss:
						logging.info('NEW BEST VALIDATION LOSS: %s, SAVING' % (val_loss))
						best_val_loss = val_loss
						if self.FLAGS.save is not None:
							self.saver.save(session, os.path.join(self.FLAGS.save, 'model.weights'))  
					logging.info('='*90) 