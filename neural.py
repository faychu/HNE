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
        self.batch_size = config.batch_size
        ############ define variables #################
        ########## MODE: 0 -- FM, 1--NFM, 2--MFM,  3--NMFM
        # MODE-0
        self.mode = config.mode
        self.struct = config.struct
        struct = self.struct
        self.h = tf.Variable(tf.ones([struct['output_dim'], 1]), trainable=False, dtype=tf.float32)
        self.V = tf.Variable(tf.random_uniform([struct['input_dim'], struct['output_dim']], -1.0, 1.0), dtype=tf.float32)
        self.w = tf.Variable(tf.random_normal([struct['input_dim'], 1]), dtype=tf.float32)
        self.w0 = tf.Variable(tf.constant(0.0), dtype=tf.float32)
        if self.mode == (1 or 3):  # neural
            print('neural')
            self.layers = len(config.struct['layers'])
            self.W = {}
            self.b = {}
            for i in range(self.layers - 1):
                name = i
                self.W[name] = tf.Variable(tf.random_normal([struct[i], struct[i + 1]]), name=name)
                self.b[name] = tf.Variable(tf.zeros([struct[i + 1]]), name=name)
        if self.mode == (2 or 3):  # textual
            print('textual')
            self.H = tf.Variable(tf.random_normal([struct['text_dim'], struct['output_dim']]))
        ###############################################

        ############## define input ###################
        # these variables are for sparse_dot
        self.X_sp_indices = tf.placeholder(tf.int64, shape=[None, 2], name='raw_indices')
        self.X_sp_val = tf.placeholder(tf.float32, shape=[None], name='raw_data')
        self.X_sp_shape = tf.placeholder(tf.int64, shape=[2], name='raw_shape')

        self.X_sp_val_ids = tf.placeholder(tf.int64, shape=[None, 2], name='val_sp_indices')
        self.X_val_shape = tf.placeholder(tf.int64, shape=[2], name='val_shape')

        self.X_sp = tf.SparseTensor(self.X_sp_indices, self.X_sp_val, self.X_sp_shape) # shape:[none,input_dim]
        self.X_val_sp = tf.SparseTensor(self.X_sp_val_ids, self.X_sp_val**2, self.X_val_shape) # shape:[none, len_v]

        self.X_indices = tf.placeholder(tf.int64, shape=[None], name='val_indices')
        if self.mode == (2 or 3):  # textual
            # Tfidf input
            self.T = tf.placeholder("float", [None, struct['text_dim']])
        ###############################################
        self.loss = self.__make_loss(config)
        self.optimizer = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)
        print("######### Finish! ########")

    def __FM(self):  # Finished!
        # factorization machine
        contribution_linear = \
            tf.sparse_tensor_dense_matmul(self.X_sp, self.w)+self.w0  # x[samples, input_dim] w[input_dim, 1]
        a = \
            (tf.sparse_tensor_dense_matmul(self.X_sp, self.V))**2  # x[samples, input_dim] V[input_dim, rank]
        V_records = tf.nn.embedding_lookup(self.V, self.X_indices)
        b = \
            tf.sparse_tensor_dense_matmul(self.X_val_sp, V_records**2)  # shape [samples,rank]
        f_v = (a-b)*0.5
        contribution_interplay = tf.matmul(f_v, self.h)  # [samples,rank],[rank,1] =[samples,1]
        output = contribution_linear + contribution_interplay #[samples,1]
        return output

    def __NFM(self):
        # neural factorization machine
        pass

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
            output = self.__NFM()
        elif config.mode == 2:
            output = self.__MFM()
        elif config.mode == 3:
            output = self.__NMFM()
        else:
            output = 0
            print("Warning: set the wrong mode!")
        loss = tf.pow(output-1.0, 2)
        return loss
        # todo

    def save_model(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def restore_model(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    def do_variables_init(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.is_Init = True
        print("############ init!! ##########")

    def __get_feed_dict(self, data):  # Finished!
        X_ind = np.array(data.X_sp_indices)
        X_val = np.array(data.X_sp_val)
        X_shape = np.array(data.X_sp_shape)
        X_val_shape = np.array(data.X_val_shape)
        X_val_ids = np.array(data.X_val_ids)
        X_indices = np.array(data.X_indices)
        return {self.X_sp_indices: X_ind, self.X_sp_shape: X_shape, self.X_sp_val: X_val,
                self.X_sp_val_ids:X_val_ids, self.X_val_shape: X_val_shape,self.X_indices:X_indices}

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
        return self.sess.run(tf.reduce_mean(self.loss), feed_dict=feed_dict)

    def get_embedding(self, data):
        return self.sess.run(self.V, feed_dict=self.__get_feed_dict(data))

    def get_W(self):
        return self.sess.run(self.W)

    def get_B(self):
        return self.sess.run(self.B)

    def close(self):
        self.sess.close()




