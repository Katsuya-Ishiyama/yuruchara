#!/usr/b:qin/python
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import yaml


# {{{ constants.
kernel1_num = 64
kernel2_num = 128
kernel3_num = 256
kernel4_num = 512
kernel5_num = 512

#kernel1_num = 2
#kernel2_num = 4
#kernel3_num = 8
#kernel4_num = 16
#kernel5_num = 16

full_connected1_node_num = 4096
full_connected2_node_num = 4096
full_connected3_node_num = 1000

response_value_num = 10

dropout_rate = 0.5
# }}}


def get_commandline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-num',
                        type=int,
                        required=True,
                        help='Number of optimaization steps.')
    return parser.parse_args()


def calculate_convolution(x, output_num):
    _shape = x.get_shape().as_list()
    input_num = _shape[-1]
    kernel = tf.random_normal(shape=[3, 3, input_num, output_num], stddev=0.2)
    _conv = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
    _relu = tf.nn.relu(_conv)
    return _relu


def calculate_max_pooling(x):
    _pool = tf.nn.max_pool(x,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    return _pool


def calculate_full_connected_layer(x, output_num):
    dim = x.get_shape().as_list()
    dim_without_batch = dim[1:]
    # TODO: 他人のプログラムから拝借したものなので、違う方法で書きたい
    flattened_dim = 1
    for d in dim_without_batch:
        flattened_dim *= d
    flattened = tf.reshape(x, shape=[-1, flattened_dim])
    w = tf.Variable(tf.truncated_normal([flattened_dim, output_num]))
    b = tf.Variable(tf.zeros(output_num))
    return tf.matmul(flattened, w) + b


args = get_commandline_args()

mnist = input_data.read_data_sets('/home/ishiyama/vgg/mnist/', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

x_image = tf.reshape(x, [-1, 28, 28, 1])

conv1_1 = calculate_convolution(x=x_image, output_num=kernel1_num)
conv1_2 = calculate_convolution(x=conv1_1, output_num=kernel1_num)
pool1 = calculate_max_pooling(conv1_2)

conv2_1 = calculate_convolution(x=pool1, output_num=kernel1_num)
conv2_2 = calculate_convolution(x=conv2_1, output_num=kernel2_num)
pool2 = calculate_max_pooling(conv2_2)

conv3_1 = calculate_convolution(x=pool2, output_num=kernel3_num)
conv3_2 = calculate_convolution(x=conv3_1, output_num=kernel3_num)
conv3_3 = calculate_convolution(x=conv3_2, output_num=kernel3_num)
conv3_4 = calculate_convolution(x=conv3_3, output_num=kernel3_num)
pool3 = calculate_max_pooling(conv3_4)

conv4_1 = calculate_convolution(x=pool3, output_num=kernel4_num)
conv4_2 = calculate_convolution(x=conv4_1, output_num=kernel4_num)
conv4_3 = calculate_convolution(x=conv4_2, output_num=kernel4_num)
conv4_4 = calculate_convolution(x=conv4_3, output_num=kernel4_num)
pool4 = calculate_max_pooling(conv4_4)

conv5_1 = calculate_convolution(x=pool4, output_num=kernel5_num)
conv5_2 = calculate_convolution(x=conv5_1, output_num=kernel5_num)
conv5_3 = calculate_convolution(x=conv5_2, output_num=kernel5_num)
conv5_4 = calculate_convolution(x=conv5_3, output_num=kernel5_num)
pool5 = calculate_max_pooling(conv5_4)

pool5 = tf.transpose(pool5, perm=[0, 3, 1, 2])

fc1 = calculate_full_connected_layer(pool5, output_num=full_connected1_node_num)
fc1 = tf.nn.dropout(fc1, dropout_rate)

fc2 = calculate_full_connected_layer(fc1, output_num=full_connected2_node_num)
fc2 = tf.nn.dropout(fc2, dropout_rate)

fc3 = calculate_full_connected_layer(fc2, output_num=full_connected3_node_num)

w0 = tf.Variable(tf.zeros([full_connected3_node_num, response_value_num]))
b0 = tf.Variable(tf.zeros([response_value_num]))
prob = tf.nn.softmax(tf.matmul(fc3, w0) + b0)

prob_dim = prob.get_shape().as_list()
t = tf.placeholder(tf.float32, [None, response_value_num])
loss = -tf.reduce_sum(t * tf.log(prob))
train = tf.train.AdamOptimizer().minimize(loss)
correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

train_num = args.train_num
images = mnist.test.images
labels = mnist.test.labels
for i in range(1, train_num+1):
    sess.run(train, feed_dict={x: images, t:labels})
    if i % 100 == 0:
        loss_val, acc_val = sess.run([loss, accuracy],
                                     feed_dict={x: images, t:labels})
        print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))

