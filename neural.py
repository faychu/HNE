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
        self.h = tf.Variable(tf.ones([struct['output_dim'], 1]), trainable=False)
        self.H = tf.Variable(tf.random_normal([struct['text_dim'], struct['layers'][0]]))
        self.V = tf.Variable(tf.random_uniform([struct['input_dim'], struct['layers'][0]], -1.0, 1.0))
        self.w = tf.Variable(tf.random_normal([struct['input_dim'], 1]))
        self.w0 = tf.Variable(0)
        for i in range(self.layers - 1):
            name = i
            self.W[name] = tf.Variable(tf.random_normal([struct[i], struct[i + 1]]), name=name)
            self.b[name] = tf.Variable(tf.zeros([struct[i + 1]]), name=name)
        ###############################################

        ############## define input ###################
        # these variables are for sparse_dot
        self.X_sp_indices = tf.placeholder(tf.int64, shape=[None, 2], name='raw_indices')
        self.X_sp_ids_val = tf.placeholder(tf.float32, shape=[None], name='raw_data')
        self.X_sp_shape = tf.placeholder(tf.int64, shape=[2], name='raw_shape')
        self.X_sp = tf.SparseTensor(self.X_sp_indices, self.X_sp_ids_val, self.X_sp_shape) # shape:[none,input_dim]
        # Tfidf input
        self.T = tf.placeholder("float", [None, struct['text_dim']])
        ###############################################
        self.loss = self.__make_loss(config)
        self.optimizer = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)

    def __FM(self):  # Finished!
        # factorization machine
        contribution_linear = \
            tf.sparse_tensor_dense_matmul(self.X_sp, self.w)+self.w0  # x[samples, input_dim] w[input_dim, 1]
        a = \
            (tf.sparse_tensor_dense_matmul(self.X_sp, self.V))**2  # x[samples, input_dim] V[input_dim, rank]
        X_split = tf.sparse_split(sp_input=self.X_sp, num_split=self.X_sp_shape[0], axis=0)
        ones = tf.ones([1, tf.shape(self.V)[1]], dtype=tf.float32)  # shape [1, rank]
        b = 0
        for sample in X_split:
            index = tf.transpose(sample.indices)[1]
            value = tf.reshape(sample.values,[-1,1])  # shape [num_selected_v,1]
            v = tf.nn.embedding_lookup(self.V, index)
            weight = tf.matmul(value, ones)  # shape [num_selected_v,rank]
            b_ = tf.reduce_sum((v * weight) ** 2,axis=0,keep_dims=True)  # shape [1,rank]
            if b == 0:
                b = b_
            else:
                b = tf.concat([b, b_], axis=0)
        # b: shape [samples,rank]
        f_v = (a-b)*0.5
        contribution_interplay = tf.matmul(f_v, self.h)  # [samples,rank],[rank,1] =[samples,1]
        output = contribution_linear - contribution_interplay
        return output

    def __MFM(self):
        # modified factorization machine
        pass

    def __NMFM(self):
        # neural modified factorization machine
        pass



    def __make_compute_graph(self):
        pass
        # todo

    def __make_loss(self, config):
        if config.mode == 0:
            output = self.__FM()
        elif config.mode == 1:
            output = self.__MFM()
        elif config.mode == 2:
            output = self.__NMFM()
        else:
            output = 0
            print("Warning: have not set the mode!")
        loss = tf.nn.l2_loss(output-1)
        return loss
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

    def __get_feed_dict(self, data):  # Finished!
        X_ind = np.array(data.X_sp_indices)
        X_val = np.array(data.X_sp_ids_val)
        X_shape = np.array(data.X_sp_shape)
        return {self.X_sp_indices: X_ind, self.X_sp_shape: X_shape, self.X_sp_ids_val: X_val}

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
        return self.sess.run(self.V)

    def get_B(self):
        return self.sess.run(self.b)

    def close(self):
        self.sess.close()




