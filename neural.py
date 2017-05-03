# coding=utf-8
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#
# File Name : neural.py
#
# Purpose : Implement the algorithm
#
# Creation Date : 02-05-2017
#
# Last Modified : Tue 2 May 2017
#
# Created By : Yunfei Chu (yfchu@bupt.edu.cn)
#
# _._._._._._._._._._._._._._._._._._._._._.
from six.moves import cPickle as pickle
import numpy as np
import tensorflow as tf
import matplotlib
import random

class NMFM:
    def __init__(self, config):
        self.is_Init = False
        self.sess = tf.Session()
        ############ define variables #################
        self.layers = len(config.struct['layers'])
        self.struct = config.struct
        struct = self.struct
        self.W = {}
        self.b = {}
        self.H = tf.Variable(tf.random_normal([struct['text_dim'], struct['layers'][0]]))
        self.V = tf.Variable(tf.random_uniform([struct['input_dim'], struct['layers'][0]], -1.0, 1.0))
        self.w = tf.Variable(tf.random_normal([struct['input_dim']+1]))
        for i in range(self.layers - 1):
            name = i
            self.W[name] = tf.Variable(tf.random_normal([struct[i], struct[i + 1]]), name=name)
            self.b[name] = tf.Variable(tf.zeros([struct[i + 1]]), name=name)
        ###############################################

        ############## define input ###################
        self.X = tf.placeholder("float", [None, struct['input_dim']])
        # these variables are for sparse_dot
        self.X_sp_indices = tf.placeholder(tf.int64, shape=[None, 2], name='raw_indices')
        self.X_sp_ids_val = tf.placeholder(tf.float32, shape=[None], name='raw_data')
        self.X_sp_shape = tf.placeholder(tf.int64, shape=[2], name='raw_shape')
        self.X_sp = tf.SparseTensor(self.X_sp_indices, self.X_sp_ids_val, self.X_sp_shape)
        # Tfidf input
        self.T = tf.placeholder("float", [None, struct['text_dim']])
        ###############################################
        self.__make_compute_graph()
        self.loss = self.__make_loss(config)
        self.optimizer = tf.train.RMSPropOptimizer(config.learning_rate).minimize(self.loss)

    def __make_compute_graph(self):
        pass
        # todo

    def __make_loss(self, config):
        def get_reg_loss(output):
            ret = tf.nn.l2_loss(output-1)
            return ret
        pass
        # todo

    def save_model(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def restore_model(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        self.is_Init = True

    def do_variables_init(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def __get_feed_dict(self, data):
        return {self.X: data.X}  # todo

    def fit(self, data):
        if (not self.is_Init):
            print("Warning: the model isn't initialized, and will be initialized randomly")
            self.do_variables_init()
        feed_dict = self.__get_feed_dict(data)
        _ = self.sess.run(self.optimizer, feed_dict=feed_dict)

    def get_loss(self, data):
        if (not self.is_Init):
            print
            "Warning: the model isn't initialized, and will be initialized randomly"
            self.do_variables_init()
        feed_dict = self.__get_feed_dict(data)
        return self.sess.run(self.loss, feed_dict=feed_dict)

    def get_embedding(self, data):
        return self.sess.run(self.H, feed_dict=self.__get_feed_dict(data))

    def get_W(self):
        return self.sess.run(self.W)

    def get_B(self):
        return self.sess.run(self.b)

    def close(self):
        self.sess.close()




