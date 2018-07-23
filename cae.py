import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random

#-------------------------------------------------#
#
# Author : Debanjan Datta  [ddatta@vt.edu]
#
# ----------------------------------------------- #
# Simple convolutional autoencoder
# Data  : MNIST digits
# Trains a simple 2 layer Convolutional Autoencoder
# IMPORTANT :
# Model used shared weights , not used in many examples online
# ----------------------------------------------- #
# Epochs : 250
# Batch size : 32
# toggle show_loss
# ------------------------------------------------#

# Config
show_loss = True
epochs = 250
batch_size = 32

class model:
	def __init__(self):
		global show_loss
		self.show_loss = show_loss
		self.set_hyperparams()
		self.mode = 'train'
		self.build()
		return None

	def set_hyperparams(self):
		global batch_size
		global epochs
		self.batch_size = batch_size
		self.num_epochs = epochs
		self.kernel_size = {
			0 : [3, 2],
			1 : [5, 3]
		}
		self.inp_channels = [1, 16]
		self.num_filters = [16, 32]
		self.strides = [
			[1, 1, 1, 1],
			[1, 1, 1, 1]
		]
		return


	def get_weight_variable(self, shape):
		with tf.name_scope('weights'):
			initial = tf.truncated_normal(shape, stddev=0.1)
			return tf.Variable(initial)

	def build(self):
		with tf.variable_scope('model'):
			self.x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
			print(type(self.x))
			x = self.x
			conv_w = []
			conv_b = []
			for i in range(2):
				dim = [
					self.kernel_size[i][0],
					self.kernel_size[i][1],
					self.inp_channels[i],
					self.num_filters[i]
				]
				w_i = self.get_weight_variable(dim)
				conv_w.append(w_i)
				dim = [self.num_filters[i]]
				b_i = self.get_weight_variable(dim)
				conv_b.append(b_i)

			# Encoder
			h_conv1 = tf.nn.conv2d(
				x,
				conv_w[0],
				strides=self.strides[0],
				padding='SAME'
			) + conv_b[0]
			conv1 = tf.nn.relu(h_conv1)

			h_conv2 = tf.nn.conv2d(
				conv1,
				conv_w[1],
				strides=self.strides[1],
				padding="SAME"
			) + conv_b[1]
			conv2 = tf.nn.relu(h_conv2)

			# Decoder
			s = self.strides[1]
			dim_0 = self.batch_size
			output_shape = (dim_0, 28, 28, 16)
			dec_1 = tf.nn.conv2d_transpose(
					value=conv2,
					filter = conv_w[1],
					output_shape = output_shape,
					strides = s,
					padding="SAME"
				)

			s = self.strides[0]
			output_shape = [dim_0, 28, 28, 1]
			dec_2 = tf.nn.conv2d_transpose(
					dec_1,
					conv_w[0],
					output_shape=output_shape,
					strides=s,
					padding='SAME'
			)

			self.result = dec_2

			_x = tf.layers.flatten(x)
			_y = tf.layers.flatten(dec_2)
			print ('_x , _y ', _x.shape , _y.shape )
			self.loss = tf.losses.mean_squared_error(_x,_y)
			self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
			self.train = self.optimizer.minimize(self.loss)
			return

	def model_train(self):
		self.mode = 'train'
		mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
		self.sess = tf.InteractiveSession()
		self.init = tf.global_variables_initializer()
		self.sess.run(self.init)
		images = mnist.test.images[0:2048]
		num_batches = images.shape[0]//self.batch_size
		bs = self.batch_size
		losses = []
		for epoch in range(self.num_epochs):
			_loss = []
			for i in range(num_batches) :
				b_images = mnist.test.images[i*bs:(i+1)*bs]
				b_images = np.reshape(b_images, [-1, 28, 28, 1])
				loss , _ = self.sess.run(
					[self.loss , self.train ],
					feed_dict={
						self.x: b_images
					})
				_loss.append(loss)
			_loss = np.mean(_loss)
			if epoch %5 == 0:
				print (_loss)
			losses.append(_loss)

		if self.show_loss == True :
			plt.plot(range(len(losses)),losses,'r-')
			plt.show()
		return

	def test_model(self , num_samples = 3):
		mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
		img = mnist.test.images[0:self.batch_size]
		img = np.reshape(img,[-1,28,28,1])
		res = self.sess.run(
			self.result,
			feed_dict={
				self.x: img
			}
		)
		for s in range(num_samples):
			idx = random.randint(0,self.batch_size)
			fig = plt.figure(1, figsize=(35, 15))
			n_columns = 2
			n_rows = 1
			plt.title('Actual vs Reconstructed')
			plt.subplot(n_rows, n_columns, 1)
			plt.imshow(img[idx, :, :, 0], interpolation="nearest", cmap="gray")
			plt.subplot(n_rows, n_columns, 2)
			plt.imshow(res[idx, :, :, 0], interpolation="nearest", cmap="gray")
			plt.show()
			path = './results/'+'img_' + str(s) + '.png'
			fig.savefig(path)
		return

# ----------------------------------------------- #

m = model()
m.model_train()
m.test_model(num_samples = 3)