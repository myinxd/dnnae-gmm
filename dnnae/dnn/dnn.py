# Copyright (C) 2018 zhixian MA <zx@mazhixian.me>
# MIT liscence

'''
A deep neural network class, may be more general.

Characters
==========
1. selectable loss function
2. configurable with configurations
3. extendable to be an auto-encoder
'''

import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm

from dnnae.utils import utils


class dnn():
    """A deep neural network class.

    inputs
    ======
    configs: object class
        configurations for the dnn
        configs.inputs: placeholder of the network's input,
                       whose shape is (None, rows, cols) or (rows*cols).
        configs.output: placeholder of the network's output
        configs.layers: a list of the hidden layers
        configs.actfun: a list of activation functions for the layers
        configs.batchflag: a list of batchly normalization flags for the layers
        configs.ae_flag: a flag of whether it is a autoencoder
        configs.share_flag: a flag of whether share the weights
        configs.keep_rate: keep rate for training the network
        configs.init_lr: initialized learning rate
        configs.numclass: number of classes to be classified

    methods
    =======
    cnn_build: build the network
    cnn_train: train the network
    cnn_test: test the network
    get_batch: get training batches
    """

    def __init__(self, configs):
        """Initializer"""
        self.inputs = configs.inputs
        self.outputs = configs.outputs
        self.labels = configs.labels
        self.ae_flag = configs.ae_flag
        self.share_flag = configs.share_flag
        self.keep_prob = configs.keep_prob
        self.numclass = configs.numclass
        # get input shape
        self.input_shape = self.inputs.get_shape().as_list()
        if len(self.input_shape) == 4:
            self.outlayer = self.input_shape[1]*self.input_shape[2]
            self.net = tf.reshape(
                self.inputs,
                [-1,
                 self.input_shape[1]*self.input_shape[2]])
        elif len(self.input_shape) == 2:
            self.outlayer = self.input_shape[1]
            self.net = self.inputs
        else:
            print("Something wrong for the input shape.")
        # AE flag
        if not self.ae_flag:
            self.layers = configs.layers
            self.actfun = configs.actfun
            self.batchflag = configs.batchflag
        else:
            # auto encoder
            self.layers = configs.layers[0:-1]
            self.actfun = configs.actfun[0:-1]
            self.batchflag = configs.batchflag[0:-1]
            self.en_layer = configs.layers[-1]
            self.en_actfun = configs.actfun[-1]
            self.en_batchflag = configs.batchflag[-1]
        # batch normalization
        self.is_training = tf.placeholder(tf.bool, name='is_training')


    def dnn_build(self):
        """Build the network"""
        self.netprinter = []
        self.netprinter.append(["Input layer",
                                self.net.get_shape().as_list()])
        # The encoder part
        with tf.name_scope("dnn"):
            with tf.name_scope("fc_en"):
                for i, layer in enumerate(self.layers):
                    self.net = fully_connected(
                        inputs=self.net,
                        num_outputs = layer,
                        activation_fn = self.actfun[i])
                    self.netprinter.append(
                        ["FC layer " + str(i), self.net.get_shape().as_list()])
                    # batch normalization
                    if self.batchflag[i]:
                        self.net = batch_norm(
                            inputs=self.net,
                            center=True,
                            scale=True,
                            is_training=self.is_training)
                        self.netprinter.append(
                            ["BN layer " + str(i), self.net.get_shape().as_list()])
                    else:
                        # dropout
                        self.net = tf.nn.dropout(
                            x=self.net,
                            keep_prob=self.keep_prob,
                            name="drop_"+str(i)
                        )
                        self.netprinter.append(
                            ["Dropout layer " + str(i), self.net.get_shape().as_list()])
            # softmax
            self.y = fully_connected(
                inputs=self.net,
                num_outputs=self.numclass,
                activation_fn = None)
            self.netprinter.append(["Softmax layer", self.y.get_shape().as_list()])
            # The decoder part optional
            if self.ae_flag:
                with tf.name_scope("en"):
                    self.net = fully_connected(
                        inputs=self.net,
                        num_outputs=self.en_layer,
                        activation_fn = self.en_actfun)
                    self.netprinter.append(["Encoder layer", self.net.get_shape().as_list()])
                    self.code = self.net
                # the decoder
                with tf.name_scope("fc_de"):
                    for i in range(len(self.layers)-1,-1,-1):
                        self.net = fully_connected(
                            inputs=self.net,
                            num_outputs = self.layers[i],
                            activation_fn = self.actfun[i])
                        self.netprinter.append(
                            ["FC layer " + str(i), self.net.get_shape().as_list()])
                        # batch normalization
                        if self.batchflag[i]:
                            self.net = batch_norm(
                                inputs=self.net,
                                center=True,
                                scale=True,
                                is_training=self.is_training)
                            self.netprinter.append(
                                ["BN layer " + str(i), self.net.get_shape().as_list()])
                        else:
                            # dropout
                            self.net = tf.nn.dropout(
                                x=self.net,
                                keep_prob=self.keep_prob,
                                name="drop_"+str(i)
                            )
                            self.netprinter.append(
                                ["Dropout layer " + str(i), self.net.get_shape().as_list()])
                # Output
                self.net = fully_connected(
                    inputs=self.net,
                    num_outputs = self.outlayer,
                    activation_fn = tf.nn.relu)
                self.netprinter.append(
                    ["Output layer", self.net.get_shape().as_list()]
                )
                if len(self.input_shape)==2:
                    self.outputs_de = self.net
                else:
                    self.outputs_de = tf.reshape(
                        self.net,
                        [-1,
                        self.input_shape[1],
                        self.input_shape[2],
                        self.input_shape[3]]
                        )


    def dnn_print(self):
        """Print the network"""
        print("Layer ID    Layer type    Layer shape")
        for i, l in enumerate(self.netprinter):
            print(i, l[0], l[1])


    def get_loss(self):
        """Get loss function"""
        with tf.name_scope("loss"):
            self.cost_ce = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.labels, logits=self.y))
            if self.ae_flag:
                self.cost_mse = tf.reduce_mean(
                    tf.square(self.outputs - self.outputs_de))


    def get_accuracy(self):
        """Get the accuracy"""
        with tf.name_scope("accuracy"):
            with tf.name_scope("correct_prediction"):
                correct_prediction = tf.equal(
                    tf.argmax(self.y, 1),
                    tf.argmax(self.labels, 1))
            with tf.name_scope("accuracy"):
                self.accuracy = tf.reduce_mean(
                    tf.cast(correct_prediction, tf.float32))


    def get_opt_ce(self):
        """Training option for cross entropy loss."""
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope("train_ops_ce"):
                self.train_op_ce = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost_ce)


    def get_opt_mse(self):
        """Training option for mean squared error"""
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope("train_ops_mse"):
                self.train_op_mse = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost_mse)


    def get_learning_rate(self):
        """Get the exponentially decreased learning rate."""
        self.init_lr = tf.placeholder(tf.float32, name="init_lr")
        self.global_step = tf.placeholder(tf.float32, name="global_step")
        self.decay_step = tf.placeholder(tf.float32, name="decay_step")
        self.decay_rate = tf.placeholder(tf.float32, name="decay_rate")
        with tf.name_scope('learning_rate'):
            self.learning_rate = tf.train.exponential_decay(
                learning_rate=self.init_lr ,
                global_step=self.global_step,
                decay_steps=self.decay_step,
                decay_rate=self.decay_rate,
                staircase=False,
                name=None)


    def loss_save(self,savepath):
        """Save the loss and accuracy staffs"""
        import pickle
        with open(savepath, 'wb') as fp:
            pickle.dump(self.train_dict, fp)


    def dnn_test(self, data, labels):
        """Test the network"""
        test_acc, test_loss = self.sess.run(
            [self.accuracy, self.cost_mse],
            feed_dict={
                self.inputs: data,
                self.labels: labels,
                self.outputs: data,
                self.is_training: False,
                self.keep_prob: 1.0})

        return test_acc, test_loss


    def dnn_train_ce(self, data, train_configs, labels=None):
        """Train the network"""

        self.get_learning_rate()
        self.get_loss()
        self.get_accuracy()
        self.get_opt_ce()

        # Init sess
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        # get validation
        data_trn,data_val = utils.gen_validation(
            data, valrate=train_configs.valrate, label=labels)

        numbatch_trn = len(data_trn["data"]) // train_configs.batchsize
        numbatch_val = len(data_val["data"]) // train_configs.batchsize

        # print("numbatch_val ", numbatch_val)

        x_epoch = np.arange(0,train_configs.epochs,1)
        y_loss_trn = np.zeros(x_epoch.shape)
        y_loss_val = np.zeros(x_epoch.shape)
        y_acc_trn = np.zeros(x_epoch.shape)
        y_acc_val = np.zeros(x_epoch.shape)

        # Init all variables
        timestamp = utils.get_timestamp()
        print("[%s]: Epochs    Trn_loss    Val_loss    Trn_acc    Val_acc" % (timestamp))
        for i in range(train_configs.epochs):
            lr_dict = {self.init_lr: train_configs.lr_init,
                       self.global_step: i,
                       self.decay_step: train_configs.batchsize,
                       self.decay_rate: train_configs.decay_rate}

            loss_trn_all = 0.0
            loss_val_all = 0.0
            acc_trn_all = 0.0
            acc_val_all = 0.0

            indices_trn = utils.gen_BatchIterator_label(
                data_trn['data'],
                data_trn['label'],
                batch_size=train_configs.batchsize,
                shuffle=True)

            for i_trn in range(numbatch_trn):
                idx_trn = indices_trn[i_trn*train_configs.batchsize:
                                      (i_trn+1)*train_configs.batchsize]
                train_dict = {
                    self.inputs: data_trn['data'][idx_trn],
                    self.labels: data_trn['label'][idx_trn],
                    self.is_training: True,
                    self.keep_prob: train_configs.keep_prob}
                train_dict.update(lr_dict)
                # train
                _, loss_trn, acc_trn = self.sess.run(
                    [self.train_op_ce, self.cost_ce, self.accuracy],
                    feed_dict=train_dict)
                loss_trn_all += loss_trn
                acc_trn_all += acc_trn

            y_loss_trn[i] = loss_trn_all / numbatch_trn
            y_acc_trn[i] = acc_trn_all / numbatch_trn

            # validation
            indices_val = utils.gen_BatchIterator_label(
                data_val['data'],
                data_val['label'],
                batch_size=train_configs.batchsize,
                shuffle=True)

            for i_val in range(numbatch_val):
                idx_val = indices_val[i_val*train_configs.batchsize:
                                      (i_val+1)*train_configs.batchsize]
                val_dict = {
                    self.inputs: data_val['data'][idx_val],
                    self.labels: data_val['label'][idx_val],
                    self.is_training: False,
                    self.keep_prob: 1.0}
                loss_val, acc_val = self.sess.run(
                    [self.cost_ce, self.accuracy],
                    feed_dict=val_dict)
                loss_val_all += loss_val
                acc_val_all += acc_val

            y_loss_val[i] = loss_val_all / numbatch_val
            y_acc_val[i] = acc_val_all / numbatch_val

            # print results
            if i % 5 == 0:
                timestamp = utils.get_timestamp()
                print('[%s]: %d    %.8f    %.8f    %.4f    %.4f' % (
                    timestamp, i,
                    y_loss_trn[i], y_loss_val[i],
                    y_acc_trn[i], y_acc_val[i]))

        self.train_dict = {
            "epochs": x_epoch,
            "trn_loss_ce": y_loss_trn,
            "val_loss_ce": y_loss_val,
            "trn_acc_ce": y_acc_trn,
            "val_acc_ce": y_acc_val}


    def dnn_train_mse(self, data, train_configs, labels=None):
        """Train the network on MNIST data"""

        self.get_learning_rate()
        self.get_loss()
        self.get_accuracy()
        self.get_opt_mse()

        # Init sess
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        # get validation
        data_trn,data_val = utils.gen_validation(
            data, valrate=train_configs.valrate, label=labels)

        numbatch_trn = len(data_trn["data"]) // train_configs.batchsize
        numbatch_val = len(data_val["data"]) // train_configs.batchsize

        x_epoch = np.arange(0,train_configs.epochs,1)
        y_loss_trn = np.zeros(x_epoch.shape)
        y_loss_val = np.zeros(x_epoch.shape)
        y_acc_trn = np.zeros(x_epoch.shape)
        y_acc_val = np.zeros(x_epoch.shape)

        # Init all variables
        timestamp = utils.get_timestamp()
        print("[%s]: Epochs    Trn_loss    Val_loss" % (timestamp))
        for i in range(train_configs.epochs):
            lr_dict = {self.init_lr: train_configs.lr_init,
                       self.global_step: i,
                       self.decay_step: train_configs.batchsize,
                       self.decay_rate: train_configs.decay_rate}

            loss_trn_all = 0.0
            loss_val_all = 0.0

            indices_trn = utils.gen_BatchIterator_label(
                data_trn['data'],
                data_trn['label'],
                batch_size=train_configs.batchsize,
                shuffle=True)

            for i_trn in range(numbatch_trn):
                idx_trn = indices_trn[i_trn*train_configs.batchsize:
                                      (i_trn+1)*train_configs.batchsize]
                train_dict = {
                    self.inputs: data_trn['data'][idx_trn],
                    self.labels: data_trn['label'][idx_trn],
                    self.outputs: data_trn['data'][idx_trn],
                    self.is_training: True,
                    self.keep_prob: train_configs.keep_prob}
                train_dict.update(lr_dict)
                # train
                _, loss_trn = self.sess.run(
                    [self.train_op_mse, self.cost_mse],
                    feed_dict=train_dict)
                loss_trn_all += loss_trn

            y_loss_trn[i] = loss_trn_all / numbatch_trn

            # validation
            indices_val = utils.gen_BatchIterator_label(
                data_val['data'],
                data_val['label'],
                batch_size=train_configs.batchsize,
                shuffle=True)

            for i_val in range(numbatch_val):
                idx_val = indices_val[i_val*train_configs.batchsize:
                                      (i_val+1)*train_configs.batchsize]
                val_dict = {
                    self.inputs: data_val['data'][idx_val],
                    self.labels: data_val['label'][idx_val],
                    self.outputs: data_val['data'][idx_val],
                    self.is_training: False,
                    self.keep_prob: 1.0}
                loss_val = self.sess.run(
                    self.cost_mse,
                    feed_dict=val_dict)
                loss_val_all += loss_val

            y_loss_val[i] = loss_val_all / numbatch_val

            # print results
            if i % 5 == 0:
                timestamp = utils.get_timestamp()
                print('[%s]: %d    %.8f    %.8f' % (
                    timestamp, i,
                    y_loss_trn[i], y_loss_val[i],
                    ))

        self.train_dict = {
            "epochs": x_epoch,
            "trn_loss_mse": y_loss_trn,
            "val_loss_mse": y_loss_val}


    def dnn_train_cmb(self, data, train_configs, labels=None):
        """Train the network with combined loss functions"""

        self.get_learning_rate()
        self.get_loss()
        self.get_accuracy()
        self.get_opt_ce()
        self.get_opt_mse()

        # get validation
        data_trn,data_val = utils.gen_validation(
            data, valrate=train_configs.valrate, label=labels)

        # Init sess
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        numbatch_trn = len(data_trn["data"]) // train_configs.batchsize
        numbatch_val = len(data_val["data"]) // train_configs.batchsize

        x_epoch = np.arange(0,train_configs.epochs,1)
        y_loss_trn_m = np.zeros(x_epoch.shape)
        y_loss_val_m = np.zeros(x_epoch.shape)
        y_loss_trn = np.zeros(x_epoch.shape)
        y_loss_val = np.zeros(x_epoch.shape)
        y_acc_trn = np.zeros(x_epoch.shape)
        y_acc_val = np.zeros(x_epoch.shape)

        # Init all variables
        timestamp = utils.get_timestamp()
        print("[%s]: Epochs    Trn_loss_mse    Val_loss_mse    Trn_loss_ce    Val_loss_ce    Trn_acc    Val_acc" % (timestamp))
        for i in range(train_configs.epochs):
            lr_dict = {self.init_lr: train_configs.lr_init,
                       self.global_step: i,
                       self.decay_step: train_configs.batchsize,
                       self.decay_rate: train_configs.decay_rate}

            loss_trn_all_m = 0.0
            loss_val_all_m = 0.0
            loss_trn_all = 0.0
            loss_val_all = 0.0
            acc_trn_all = 0.0
            acc_val_all = 0.0

            indices_trn = utils.gen_BatchIterator_label(
                data_trn['data'],
                data_trn['label'],
                batch_size=train_configs.batchsize,
                shuffle=True)

            for i_trn in range(numbatch_trn):
                idx_trn = indices_trn[i_trn*train_configs.batchsize:
                                      (i_trn+1)*train_configs.batchsize]
                train_dict = {
                    self.inputs: data_trn['data'][idx_trn],
                    self.labels: data_trn['label'][idx_trn],
                    self.outputs: data_trn['data'][idx_trn],
                    self.is_training: True,
                    self.keep_prob: train_configs.keep_prob}
                train_dict.update(lr_dict)
                # train
                _, loss_trn_m = self.sess.run(
                    [self.train_op_mse, self.cost_mse],
                    feed_dict=train_dict)
                _, loss_trn, acc_trn = self.sess.run(
                    [self.train_op_ce, self.cost_ce, self.accuracy],
                    feed_dict=train_dict)
                loss_trn_all_m += loss_trn_m
                acc_trn_all += acc_trn
                loss_trn_all += loss_trn

            y_loss_trn_m[i] = loss_trn_all_m / numbatch_trn
            y_acc_trn[i] = acc_trn_all / numbatch_trn
            y_loss_trn[i] = loss_trn_all / numbatch_trn

            # validation
            indices_val = utils.gen_BatchIterator_label(
                data_val['data'],
                data_val['label'],
                batch_size=train_configs.batchsize,
                shuffle=True)

            for i_val in range(numbatch_val):
                idx_val = indices_val[i_val*train_configs.batchsize:
                                      (i_val+1)*train_configs.batchsize]
                val_dict = {
                    self.inputs: data_val['data'][idx_val],
                    self.labels: data_val['label'][idx_val],
                    self.outputs: data_val['data'][idx_val],
                    self.is_training: False,
                    self.keep_prob: 1.0}
                loss_val_m = self.sess.run(
                    self.cost_mse,
                    feed_dict=val_dict)
                loss_val, acc_val = self.sess.run(
                    [self.cost_ce, self.accuracy],
                    feed_dict=val_dict)

                loss_val_all_m += loss_val_m
                acc_val_all += acc_val
                loss_val_all += loss_val

            y_loss_val_m[i] = loss_val_all_m / numbatch_val
            y_acc_val[i] = acc_val_all / numbatch_val
            y_loss_val[i] = loss_val_all / numbatch_val

            # print results
            if i % 5 == 0:
                timestamp = utils.get_timestamp()
                print('[%s]: %d    %.8f    %.8f    %.8f    %.8f    %.4f    %.4f' % (
                    timestamp, i,
                    y_loss_trn_m[i], y_loss_val_m[i],
                    y_loss_trn[i], y_loss_val[i],
                    y_acc_trn[i], y_acc_val[i]))

        self.train_dict = {
            "epochs": x_epoch,
            "trn_loss_mse": y_loss_trn_m,
            "val_loss_mse": y_loss_val_m,
            "trn_loss_ce": y_loss_trn,
            "val_loss_ce": y_loss_val,
            "trn_acc_ce": y_acc_trn,
            "val_acc_ce": y_acc_val}



    def dnn_train_mnist(self, mnist, train_configs):
        """Train the network on MNIST data"""

        self.get_learning_rate()
        self.get_loss()
        self.get_accuracy()
        self.get_opt_ce()

        # Init sess
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        numbatch_trn = mnist.train.images.shape[0] // train_configs.batchsize
        numbatch_val = mnist.validation.images.shape[0] // train_configs.batchsize

        x_epoch = np.arange(0,train_configs.epochs,1)
        y_loss_trn = np.zeros(x_epoch.shape)
        y_loss_val = np.zeros(x_epoch.shape)
        y_acc_trn = np.zeros(x_epoch.shape)
        y_acc_val = np.zeros(x_epoch.shape)

        # Init all variables
        timestamp = utils.get_timestamp()
        print("[%s]: Epochs    Trn_loss    Val_loss    Trn_acc    Val_acc" % (timestamp))
        for i in range(train_configs.epochs):
            lr_dict = {self.init_lr: train_configs.lr_init,
                       self.global_step: i,
                       self.decay_step: train_configs.batchsize,
                       self.decay_rate: train_configs.decay_rate}

            loss_trn_all = 0.0
            loss_val_all = 0.0
            acc_trn_all = 0.0
            acc_val_all = 0.0

            for i_trn in range(numbatch_trn):
                data_trn, label_trn = mnist.train.next_batch(
                      batch_size=train_configs.batchsize)
                train_dict = {
                    self.inputs: data_trn,
                    self.labels: label_trn,
                    self.is_training: True,
                    self.keep_prob: train_configs.keep_prob}
                train_dict.update(lr_dict)
                # train
                _, loss_trn, acc_trn = self.sess.run(
                    [self.train_op_ce, self.cost_ce, self.accuracy],
                    feed_dict=train_dict)
                loss_trn_all += loss_trn
                acc_trn_all += acc_trn

            y_loss_trn[i] = loss_trn_all / numbatch_trn
            y_acc_trn[i] = acc_trn_all / numbatch_trn

            # validation
            for i_trn in range(numbatch_val):
                data_val, label_val = mnist.validation.next_batch(
                      batch_size=train_configs.batchsize)
                val_dict = {
                    self.inputs: data_val,
                    self.labels: label_val,
                    self.is_training: False,
                    self.keep_prob: 1.0}
                loss_val, acc_val = self.sess.run(
                    [self.cost_ce, self.accuracy],
                    feed_dict=val_dict)
                loss_val_all += loss_val
                acc_val_all += acc_val

            y_loss_val[i] = loss_val_all / numbatch_val
            y_acc_val[i] = acc_val_all / numbatch_val

            # print results
            if i % 5 == 0:
                timestamp = utils.get_timestamp()
                print('[%s]: %d    %.8f    %.8f    %.4f    %.4f' % (
                    timestamp, i,
                    y_loss_trn[i], y_loss_val[i],
                    y_acc_trn[i], y_acc_val[i]))

        self.train_dict = {
            "epochs": x_epoch,
            "trn_loss_ce": y_loss_trn,
            "val_loss_ce": y_loss_val,
            "trn_acc_ce": y_acc_trn,
            "val_acc_ce": y_acc_val}


    def dnn_train_mnist_mse(self, mnist, train_configs):
        """Train the network on MNIST data"""

        self.get_learning_rate()
        self.get_loss()
        self.get_accuracy()
        self.get_opt_mse()

        # Init sess
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        numbatch_trn = mnist.train.images.shape[0] // train_configs.batchsize
        numbatch_val = mnist.validation.images.shape[0] // train_configs.batchsize

        x_epoch = np.arange(0,train_configs.epochs,1)
        y_loss_trn = np.zeros(x_epoch.shape)
        y_loss_val = np.zeros(x_epoch.shape)
        y_acc_trn = np.zeros(x_epoch.shape)
        y_acc_val = np.zeros(x_epoch.shape)

        # Init all variables
        timestamp = utils.get_timestamp()
        print("[%s]: Epochs    Trn_loss    Val_loss" % (timestamp))
        for i in range(train_configs.epochs):
            lr_dict = {self.init_lr: train_configs.lr_init,
                       self.global_step: i,
                       self.decay_step: train_configs.batchsize,
                       self.decay_rate: train_configs.decay_rate}

            loss_trn_all = 0.0
            loss_val_all = 0.0

            for i_trn in range(numbatch_trn):
                data_trn, label_trn = mnist.train.next_batch(
                      batch_size=train_configs.batchsize)
                train_dict = {
                    self.inputs: data_trn,
                    self.labels: label_trn,
                    self.outputs: data_trn,
                    self.is_training: True,
                    self.keep_prob: train_configs.keep_prob}
                train_dict.update(lr_dict)
                # train
                _, loss_trn = self.sess.run(
                    [self.train_op_mse, self.cost_mse],
                    feed_dict=train_dict)
                loss_trn_all += loss_trn

            y_loss_trn[i] = loss_trn_all / numbatch_trn

            # validation
            for i_trn in range(numbatch_val):
                data_val, label_val = mnist.validation.next_batch(
                      batch_size=train_configs.batchsize)
                val_dict = {
                    self.inputs: data_val,
                    self.labels: label_val,
                    self.outputs: data_val,
                    self.is_training: False,
                    self.keep_prob: 1.0}
                loss_val = self.sess.run(
                    self.cost_mse,
                    feed_dict=val_dict)
                loss_val_all += loss_val

            y_loss_val[i] = loss_val_all / numbatch_val

            # print results
            if i % 5 == 0:
                timestamp = utils.get_timestamp()
                print('[%s]: %d    %.8f    %.8f' % (
                    timestamp, i,
                    y_loss_trn[i], y_loss_val[i],
                    ))

        self.train_dict = {
            "epochs": x_epoch,
            "trn_loss_mse": y_loss_trn,
            "val_loss_mse": y_loss_val}


    def dnn_train_mnist_cmb(self, mnist, train_configs):
        """Train the network on MNIST data"""

        self.get_learning_rate()
        self.get_loss()
        self.get_accuracy()
        self.get_opt_ce()
        self.get_opt_mse()

        # Init sess
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        numbatch_trn = mnist.train.images.shape[0] // train_configs.batchsize
        numbatch_val = mnist.validation.images.shape[0] // train_configs.batchsize

        x_epoch = np.arange(0,train_configs.epochs,1)
        y_loss_trn_m = np.zeros(x_epoch.shape)
        y_loss_val_m = np.zeros(x_epoch.shape)
        y_loss_trn = np.zeros(x_epoch.shape)
        y_loss_val = np.zeros(x_epoch.shape)
        y_acc_trn = np.zeros(x_epoch.shape)
        y_acc_val = np.zeros(x_epoch.shape)

        # Init all variables
        timestamp = utils.get_timestamp()
        print("[%s]: Epochs    Trn_loss_mse    Val_loss_mse    Trn_loss_ce    Val_loss_ce    Trn_acc    Val_acc" % (timestamp))
        for i in range(train_configs.epochs):
            lr_dict = {self.init_lr: train_configs.lr_init,
                       self.global_step: i,
                       self.decay_step: train_configs.batchsize,
                       self.decay_rate: train_configs.decay_rate}

            loss_trn_all_m = 0.0
            loss_val_all_m = 0.0
            loss_trn_all = 0.0
            loss_val_all = 0.0
            acc_trn_all = 0.0
            acc_val_all = 0.0


            for i_trn in range(numbatch_trn):
                data_trn, label_trn = mnist.train.next_batch(
                      batch_size=train_configs.batchsize)
                train_dict = {
                    self.inputs: data_trn,
                    self.labels: label_trn,
                    self.outputs: data_trn,
                    self.is_training: True,
                    self.keep_prob: train_configs.keep_prob}
                train_dict.update(lr_dict)
                # train
                _, loss_trn_m = self.sess.run(
                    [self.train_op_mse, self.cost_mse],
                    feed_dict=train_dict)
                _, loss_trn, acc_trn = self.sess.run(
                    [self.train_op_ce, self.cost_ce, self.accuracy],
                    feed_dict=train_dict)
                loss_trn_all_m += loss_trn_m
                acc_trn_all += acc_trn
                loss_trn_all += loss_trn

            y_loss_trn_m[i] = loss_trn_all_m / numbatch_trn
            y_acc_trn[i] = acc_trn_all / numbatch_trn
            y_loss_trn[i] = loss_trn_all / numbatch_trn

            # validation
            for i_trn in range(numbatch_val):
                data_val, label_val = mnist.validation.next_batch(
                      batch_size=train_configs.batchsize)
                val_dict = {
                    self.inputs: data_val,
                    self.labels: label_val,
                    self.outputs: data_val,
                    self.is_training: False,
                    self.keep_prob: 1.0}
                loss_val_m = self.sess.run(
                    self.cost_mse,
                    feed_dict=val_dict)
                loss_val, acc_val = self.sess.run(
                    [self.cost_ce, self.accuracy],
                    feed_dict=val_dict)

                loss_val_all_m += loss_val_m
                acc_val_all += acc_val
                loss_val_all += loss_val

            y_loss_val_m[i] = loss_val_all_m / numbatch_val
            y_acc_val[i] = acc_val_all / numbatch_val
            y_loss_val[i] = loss_val_all / numbatch_val

            # print results
            if i % 5 == 0:
                timestamp = utils.get_timestamp()
                print('[%s]: %d    %.8f    %.8f    %.8f    %.8f    %.4f    %.4f' % (
                    timestamp, i,
                    y_loss_trn_m[i], y_loss_val_m[i],
                    y_loss_trn[i], y_loss_val[i],
                    y_acc_trn[i], y_acc_val[i]))

        self.train_dict = {
            "epochs": x_epoch,
            "trn_loss_mse": y_loss_trn_m,
            "val_loss_mse": y_loss_val_m,
            "trn_loss_ce": y_loss_trn,
            "val_loss_ce": y_loss_val,
            "trn_acc_ce": y_acc_trn,
            "val_acc_ce": y_acc_val}
