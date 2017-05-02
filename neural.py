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

class NFM:
    def __init__(self, config):

        ############ define variables ##################
        self.layers = len(config.struct)
        self.struct = config.struct
        self.sparse_dot = config.sparse_dot
        self.W = {}
        self.b = {}
        struct = self.struct
        for i in range(self.layers - 1):
            name = "encoder" + str(i)
            self.W[name] = tf.Variable(tf.random_normal([struct[i], struct[i + 1]]), name=name)
            self.b[name] = tf.Variable(tf.zeros([struct[i + 1]]), name=name)
        struct.reverse()
        for i in range(self.layers - 1):
            name = "decoder" + str(i)
            self.W[name] = tf.Variable(tf.random_normal([struct[i], struct[i + 1]]), name=name)
            self.b[name] = tf.Variable(tf.zeros([struct[i + 1]]), name=name)
        self.struct.reverse()
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
