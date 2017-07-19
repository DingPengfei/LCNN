# -*- coding:utf-8 -*-

import os
import numpy as np
import pandas as pd
import scipy.io as sio
import tensorflow as tf

np.random.seed(10)
Dimension = 1900
single_sample = 560
Samples = 44 * 560 + 29 # 24669

totalIter = 10000+1

trainX = np.zeros([ Samples - single_sample, 8, Dimension ])
trainY = np.zeros([ Samples - single_sample, 1 ])
testX = np.zeros([ single_sample, 8, Dimension ])
testY = np.zeros([ single_sample, 1 ])


conv1_size = 21; conv1_feature = 6; max_pool1_size = 7
conv2_size = 13; conv2_feature = 7; max_pool2_size = 6
conv3_size =  9; conv3_feature = 5; max_pool3_size = 6

fc_connect_size = 50


def ReadDataSet():
	global trainX, trainY, testX, testY

	X = np.zeros([ Samples, 8, Dimension ])
	Label = np.zeros([ Samples, 1 ])

	path = 'dataset/'
	for i in range(44):
		filename = 'jinshanwhole' + str(i).zfill(2) + '.mat'
		
		# ecgBeatsInfo shape : 560 * 15205
		# 15205 = 0~0 (order) + 
		# 1~1 (constant：1) + 
		# 2~2 (constant: 8) + 
		# 3~3 (constant: 1900) +
		# 4~4 (classification 0:Normal 1:Abnormal) +
		# 5~15204 = 8(II, III, V1, V2, V3, V4, V5, V6) * 1900

		mat = sio.loadmat(path + filename)['ecgBeatsInfo']
		# block_size = len(mat)	# 0~43:560, 44:29
		
		st = i * 560; en = st
		if (i< 44):
			en += 560
		else:
			en += 29

		X[st:en] = mat[:, 5:].reshape(-1, 8, Dimension)	# block_size * 8 * 1900
		Label[st:en] = mat[:, 4].reshape(-1, 1)	# block_size * 1

	'''
	# memory overflow
	mean = X.mean(axis=2); std = X.std(axis=2)
	X = (X - mean[:,:,np.newaxis])/(std[:,:,np.newaxis] + 1e-6)
	'''
	print ("shape of X:", X.shape)
	'''
	mean = X.mean(axis=2); std = X.std(axis=2)
	for i in range(np.shape(X)[0]):
		for j in range(np.shape(X)[1]):
			for k in range(np.shape(X)[2]):
				X[i][j][k] = (X[i][j][k] - mean[i][j]) / (std[i][j] + 1e-6)
	'''

	# split the dataset
	trainX = X[single_sample:, :, :]
	trainY = Label[single_sample:, :]
	
	testX = X[:single_sample, :, :Dimension-200]
	testY = Label[:single_sample, :]

	print ("Finish reading.")


def set_varibale_weight(layer):
	# shape : [heights, width, in_channels, out_channels]
	if layer=='conv1':
		init = tf.Variable( tf.truncated_normal( shape=[1, conv1_size, 1, conv1_feature], stddev=0.01 ) )
	elif layer=='conv2':
		init = tf.Variable( tf.truncated_normal( shape=[1, conv2_size, conv1_feature, conv2_feature], stddev=0.01 ) )
	else:
		init = tf.Variable( tf.truncated_normal( shape=[1, conv3_size, conv2_feature, conv3_feature], stddev=0.01 ) )
	return init


def set_variable_bias(layer):
	if layer=='conv1':
		init = tf.Variable( tf.constant(0.01, shape=[conv1_feature]) )
	elif layer=='conv2':
		init = tf.Variable( tf.constant(0.01, shape=[conv2_feature]) )
	else:
		init = tf.Variable( tf.constant(0.01, shape=[conv3_feature]) )
	return init


def conv2d(X, w):
	return tf.nn.conv2d( X, w, strides = [1,1,1,1], padding='VALID' )


def relu(X):
	return tf.nn.relu(X)


def max_pool(relu_x, layer):
	if layer==1:
		res = tf.nn.max_pool( relu_x, ksize = [1, 1, max_pool1_size, 1], strides = [1, 1, max_pool1_size, 1], padding = 'VALID' )
	else:
		res = tf.nn.max_pool( relu_x, ksize = [1, 1, max_pool2_size, 1], strides = [1, 1, max_pool2_size, 1], padding = 'VALID' )
	return res

def next_batch(batch_size=128):
	# should be integer of 0~42
	index = np.random.randint(43)
	# print ("train set is ", i)
	batch_xs = np.zeros([batch_size, 8, Dimension-200, 1])
	batch_ys = np.zeros([batch_size, 1])

	for i in range(batch_size):
		type = np.random.randint(2)
		if type == 1:
			# 1 ~ 160
			x = np.random.randint(160)+1
			start = np.random.randint(200)

			batch_xs[i] = trainX[ index*560+x, :, start:start+Dimension-200 ].reshape(-1, 8, Dimension-200, 1)
			batch_ys[i] = trainY[ index*560+x ]
		else:
			# 161 ~ 560
			x = np.random.randint(400)+161
			start = np.random.randint(200)

			batch_xs[i] = trainX[ index*560+x, :, start:start+Dimension-200 ].reshape(-1, 8, Dimension-200, 1)
			batch_ys[i] = trainY[ index*560+x ]

	return batch_xs, batch_ys


def BuildConv1d():

	tf_X = tf.placeholder( tf.float32, [None, 8, Dimension-200, 1] )
	# 8 Leader Input
	tf_X_II = tf_X[:, 0:1, :, :]
	tf_X_III =tf_X[:, 1:2, :, :]
	tf_X_V1 = tf_X[:, 2:3, :, :]
	tf_X_V2 = tf_X[:, 3:4, :, :]
	tf_X_V3 = tf_X[:, 4:5, :, :]
	tf_X_V4 = tf_X[:, 5:6, :, :]
	tf_X_V5 = tf_X[:, 6:7, :, :]
	tf_X_V6 = tf_X[:, 7:8, :, :]
	print (tf_X_II)	# shape = (?, 1, 1700, 1)

	# label
	tf_Y = tf.placeholder( tf.float32, [None, 1] )

	# conv1
	II_conv_w1 = set_varibale_weight('conv1'); II_conv_b1 = set_variable_bias('conv1')
	III_conv_w1 =set_varibale_weight('conv1'); III_conv_b1 =set_variable_bias('conv1')
	V1_conv_w1 = set_varibale_weight('conv1'); V1_conv_b1 = set_variable_bias('conv1')
	V2_conv_w1 = set_varibale_weight('conv1'); V2_conv_b1 = set_variable_bias('conv1')
	V3_conv_w1 = set_varibale_weight('conv1'); V3_conv_b1 = set_variable_bias('conv1')
	V4_conv_w1 = set_varibale_weight('conv1'); V4_conv_b1 = set_variable_bias('conv1')
	V5_conv_w1 = set_varibale_weight('conv1'); V5_conv_b1 = set_variable_bias('conv1')
	V6_conv_w1 = set_varibale_weight('conv1'); V6_conv_b1 = set_variable_bias('conv1')

	# relu1 + max_pool1
	II_maxpool1 = max_pool( relu( conv2d(tf_X_II, II_conv_w1) + II_conv_b1 ), 1 )
	III_maxpool1 =max_pool( relu( conv2d(tf_X_III, III_conv_w1) + III_conv_b1 ), 1 )
	V1_maxpool1 = max_pool( relu( conv2d(tf_X_V1, V1_conv_w1) + V1_conv_b1 ), 1 )
	V2_maxpool1 = max_pool( relu( conv2d(tf_X_V2, V2_conv_w1) + V2_conv_b1 ), 1 )
	V3_maxpool1 = max_pool( relu( conv2d(tf_X_V3, V3_conv_w1) + V3_conv_b1 ), 1 )
	V4_maxpool1 = max_pool( relu( conv2d(tf_X_V4, V4_conv_w1) + V4_conv_b1 ), 1 )
	V5_maxpool1 = max_pool( relu( conv2d(tf_X_V5, V5_conv_w1) + V5_conv_b1 ), 1 )
	V6_maxpool1 = max_pool( relu( conv2d(tf_X_V6, V6_conv_w1) + V6_conv_b1 ), 1 )
	print (II_maxpool1)	# shape = (?, 1, 240, 6) 240=(1700-21+1)/7

	# conv2
	II_conv_w2 = set_varibale_weight('conv2'); II_conv_b2 = set_variable_bias('conv2')
	III_conv_w2 =set_varibale_weight('conv2'); III_conv_b2 =set_variable_bias('conv2')
	V1_conv_w2 = set_varibale_weight('conv2'); V1_conv_b2 = set_variable_bias('conv2')
	V2_conv_w2 = set_varibale_weight('conv2'); V2_conv_b2 = set_variable_bias('conv2')
	V3_conv_w2 = set_varibale_weight('conv2'); V3_conv_b2 = set_variable_bias('conv2')
	V4_conv_w2 = set_varibale_weight('conv2'); V4_conv_b2 = set_variable_bias('conv2')
	V5_conv_w2 = set_varibale_weight('conv2'); V5_conv_b2 = set_variable_bias('conv2')
	V6_conv_w2 = set_varibale_weight('conv2'); V6_conv_b2 = set_variable_bias('conv2')

	# relu2 + max_pool2
	II_maxpool2 = max_pool( relu( conv2d(II_maxpool1, II_conv_w2) + II_conv_b2 ), 2 )
	III_maxpool2 =max_pool( relu( conv2d(III_maxpool1, III_conv_w2) + III_conv_b2 ), 2 )
	V1_maxpool2 = max_pool( relu( conv2d(V1_maxpool1, V1_conv_w2) + V1_conv_b2 ), 2 )
	V2_maxpool2 = max_pool( relu( conv2d(V2_maxpool1, V2_conv_w2) + V2_conv_b2 ), 2 )
	V3_maxpool2 = max_pool( relu( conv2d(V3_maxpool1, V3_conv_w2) + V3_conv_b2 ), 2 )
	V4_maxpool2 = max_pool( relu( conv2d(V4_maxpool1, V4_conv_w2) + V4_conv_b2 ), 2 )
	V5_maxpool2 = max_pool( relu( conv2d(V5_maxpool1, V5_conv_w2) + V5_conv_b2 ), 2 )
	V6_maxpool2 = max_pool( relu( conv2d(V6_maxpool1, V6_conv_w2) + V6_conv_b2 ), 2 )
	print (II_maxpool2) # shape = (?, 1, 38, 7) 38=(240-13+1)/6

	# conv3
	II_conv_w3 = set_varibale_weight('conv3'); II_conv_b3 = set_variable_bias('conv3')
	III_conv_w3 =set_varibale_weight('conv3'); III_conv_b3 =set_variable_bias('conv3')
	V1_conv_w3 = set_varibale_weight('conv3'); V1_conv_b3 = set_variable_bias('conv3')
	V2_conv_w3 = set_varibale_weight('conv3'); V2_conv_b3 = set_variable_bias('conv3')
	V3_conv_w3 = set_varibale_weight('conv3'); V3_conv_b3 = set_variable_bias('conv3')
	V4_conv_w3 = set_varibale_weight('conv3'); V4_conv_b3 = set_variable_bias('conv3')
	V5_conv_w3 = set_varibale_weight('conv3'); V5_conv_b3 = set_variable_bias('conv3')
	V6_conv_w3 = set_varibale_weight('conv3'); V6_conv_b3 = set_variable_bias('conv3')

	# relu3 + max_pool3
	II_maxpool3 = max_pool( relu( conv2d(II_maxpool2, II_conv_w3) + II_conv_b3 ), 3 )
	III_maxpool3 =max_pool( relu( conv2d(III_maxpool2, III_conv_w3) + III_conv_b3 ), 3 )
	V1_maxpool3 = max_pool( relu( conv2d(V1_maxpool2, V1_conv_w3) + V1_conv_b3 ), 3 )
	V2_maxpool3 = max_pool( relu( conv2d(V2_maxpool2, V2_conv_w3) + V2_conv_b3 ), 3 )
	V3_maxpool3 = max_pool( relu( conv2d(V3_maxpool2, V3_conv_w3) + V3_conv_b3 ), 3 )
	V4_maxpool3 = max_pool( relu( conv2d(V4_maxpool2, V4_conv_w3) + V4_conv_b3 ), 3 )
	V5_maxpool3 = max_pool( relu( conv2d(V5_maxpool2, V5_conv_w3) + V5_conv_b3 ), 3 )
	V6_maxpool3 = max_pool( relu( conv2d(V6_maxpool2, V6_conv_w3) + V6_conv_b3 ), 3 )
	print (II_maxpool3) # shape = (?, 1, 5, 5) 5=(38-9+1)/6

	# full_connection
	# reshape per leader to [None, 25]
	fc_II = tf.reshape(II_maxpool3, [-1, 5*5])
	fc_III = tf.reshape(III_maxpool3, [-1, 5*5])
	fc_V1 = tf.reshape(V1_maxpool3, [-1, 5*5])
	fc_V2 = tf.reshape(V2_maxpool3, [-1, 5*5])
	fc_V3 = tf.reshape(V3_maxpool3, [-1, 5*5])
	fc_V4 = tf.reshape(V4_maxpool3, [-1, 5*5])
	fc_V5 = tf.reshape(V5_maxpool3, [-1, 5*5])
	fc_V6 = tf.reshape(V6_maxpool3, [-1, 5*5])

	# concat the 8 leader's feature
	fc_combine = tf.stack( [fc_II, fc_III, fc_V1, fc_V2, fc_V3, fc_V4, fc_V5, fc_V6], 1 )
	print("fc_combine:", fc_combine)	# shape = (?, 8, 25)

	fc = tf.reshape(fc_combine, [-1, 8*25])

	fc_w = tf.Variable( tf.truncated_normal([8*25, fc_connect_size], stddev=0.1) )
	fc_b = tf.Variable( tf.constant(0.1, shape=[fc_connect_size]) )
	fc_out = tf.nn.relu( tf.matmul(fc, fc_w) + fc_b )

	logistic_w = tf.Variable( tf.truncated_normal([fc_connect_size, 1], stddev=0.1) )
	logistic_b = tf.Variable( tf.constant(0.1, shape=[1]) )
	pred = tf.nn.sigmoid( tf.matmul( fc_out, logistic_w ) + logistic_b )
	print ("pred:", pred)	# shape = (?, 1)

	loss = tf.reduce_mean( - tf_Y * tf.log(tf.clip_by_value(pred,1e-10,1.0)) - (1-tf_Y) * tf.log(tf.clip_by_value(1-pred,1e-10,1.0)) )
	print (loss)	# shape = ()

	current_iter = tf.Variable(0)

	learning_rate = tf.train.exponential_decay(0.001, current_iter, decay_steps=totalIter, decay_rate=0.1)

	train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=current_iter)

	accuracy = tf.reduce_mean( tf.cast( tf.equal( tf.round(tf_Y), tf.round(pred) ) , tf.float32 ) )

	# Save the best model
	saver = tf.train.Saver(max_to_keep=1)

	# mkdir if not exists
	model_dir = "save_best_model/model1"
	model_name = "ckp"
	if not os.path.exists(model_dir):
		os.mkdir(model_dir)

	bestAcc = 0
	with tf.Session() as sess:
		sess.run( tf.global_variables_initializer() )
		for iter in range(totalIter):
			current_iter = iter
			batch_xs, batch_ys = next_batch(batch_size=512)
			_, acc, loss_val, lr, pred_train  = sess.run( [train_op, accuracy, loss, learning_rate, pred], feed_dict={tf_X : batch_xs, tf_Y : batch_ys} )
			print ("After %d iter, lr is : %g, loss is : %g, train accuracy is : %g" %(iter, lr, loss_val, acc) )
			# print (np.round(pred_train.reshape(1, -1)))

			if (iter % 10 == 0):
				# run the validation dataset
				batch_testx = testX[:,:,:Dimension-200].reshape(-1,8,Dimension-200,1)
				_, acc_test, pred_test = sess.run( [train_op, accuracy, pred], feed_dict={tf_X : batch_testx, tf_Y : testY} )
				print ("After %d iter, test accuracy is : %g" %(iter, acc_test) )
				print (np.round(pred_test.reshape(1, -1)))
				if (acc_test > bestAcc):
					bestAcc = acc_test
					saver.save(sess, os.path.join(model_dir, model_name+str(iter)+"#"+str(acc_test)))
					'''
					# run the small-scale dataset
					batch_small_scale_x = trainX[:, :, :Dimension-200].reshape(-1, 8, Dimension-200, 1)
					batch_small_scale_y = trainY
					_, acc_small_scale, pred_small_scale = sess.run( [train_op, accuracy, pred], feed_dict = {tf_X : batch_small_scale_x, tf_Y : batch_small_scale_y} )
					print ("small-scale accuracy is : %g" % acc_small_scale )
					print (np.round(pred_small_scale.reshape(1, -1)))
					'''


if __name__ == '__main__':
	ReadDataSet()
	BuildConv1d()
