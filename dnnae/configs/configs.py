# Copyright (C) 2018 Zhixian MA <zx@mazhixian.me>
# MIT liscence

"""Configurations for the DNNAE network."""

import tensorflow as tf

class config_mnist_bn(object):
    rs = 28
    inputs = tf.placeholder(dtype=tf.float32, shape=(None,rs**2), name='x_in')
    outputs = tf.placeholder(dtype=tf.float32, shape=(None,rs**2), name='x_out')
    numclass = 10
    labels = tf.placeholder(dtype=tf.float32, shape=(None,numclass), name='labels')
    ae_flag = True
    share_flag = False
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    layers = [256, 128, 32]
    actfun = [tf.nn.relu, tf.nn.relu, tf.nn.relu]
    batchflag = [True, True, False]


class config_mnist_do(object):
    rs = 28
    inputs = tf.placeholder(dtype=tf.float32, shape=(None,rs**2), name='x_in')
    outputs = tf.placeholder(dtype=tf.float32, shape=(None,rs**2), name='x_out')
    numclass = 10
    labels = tf.placeholder(dtype=tf.float32, shape=(None,numclass), name='labels')
    ae_flag = True
    share_flag = False
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    layers = [256, 128, 32]
    actfun = [tf.nn.relu, tf.nn.relu, tf.nn.relu]
    batchflag = [False, False, False]


class config_train(object):
    valrate = 0.2
    batchsize = 100
    epochs = 100
    lr_init = 0.0001
    decay_rate = 0.95
    keep_prob = 0.5
