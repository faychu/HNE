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
        self.X = tf.placeholder("float", [None, struct['input_dim']])  # todo
        self.T = tf.placeholder("float", [None, struct['text_dim']])
        ###############################################

# Step 1: load data
# Step  : Build and train model
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)  # 变量的初始值为截断正太分布
    return tf.Variable(initial, dtype=tf.float32)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Define placeholders
