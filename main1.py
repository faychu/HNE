#!/usr/bin/python2
# -*- coding: utf-8 -*-
# coding=utf-8
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#
# File Name : data.py
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


from config import Config
from data import Records
from neural import NMFM
import numpy as np
from utils.utils import *
import time
import scipy.io as sio

if __name__ == "__main__":
    config = Config()

    records_data = Records(config.file_path)
    config.struct['input_dim'] = records_data.num_nodes
    # print(records_data.records[0:10])
    # print(records_data.X_sp_indices[0:10])
    # print(records_data.X_sp_val[0:10])

    # for i in range(10):
    #     mini_batch = records_data.sample(config.batch_size,do_shuffle=False)
    #     # print(mini_batch.records)
    #     # print(mini_batch.X_sp_indices)
    #     # print(mini_batch.X_indices)
    #     # print(mini_batch.X_sp_val)
    #     # print(mini_batch.X_val_ids)
    print('########### START ##########')
    model = NMFM(config)
    print('############model###########')
    model.do_variables_init()


    #
    last_loss = np.inf  # 无限大的正数
    converge_count = 0
    time_consumed = 0
    epochs = 0
    loss = 0
    # mini_batch = records_data.sample(config.batch_size,do_shuffle=False)
    # print(mini_batch.records)
    # print(mini_batch.X_sp_indices)
    # print(mini_batch.X_indices)
    # print(mini_batch.X_sp_val)
    # print(mini_batch.X_val_ids)
    mini_batch = Dotdict()
    mini_batch.X_sp_indices = np.array([[0, 0], [0, 1], [0, 2],[1,3],[1,4],[1,5]])
    mini_batch.X_indices = np.array([0,1,2,3,4,5])
    mini_batch.X_sp_val = np.array([1.0,0.9,0.8,0.7,0.6,0.5])
    mini_batch.X_val_ids = np.array([[0, 0], [0, 1], [0, 2],[1,3],[1,4],[1,5]])
    mini_batch.X_sp_shape = [2, 2085231]
    mini_batch.X_val_shape = [2, 6]
    for i in range(15):
        # mini_batch = records_data.sample(config.batch_size)
        st_time = time.time()
        model.fit(mini_batch)
        time_consumed += time.time() - st_time
        loss = model.get_loss(mini_batch)
        output = model.get_output(mini_batch)
        embedding = model.get_embedding(mini_batch)
        a = model.get_a(mini_batch)
        b = model.get_b(mini_batch)
        w0 = model.get_w0(mini_batch)
        w = model.get_w(mini_batch)
        print("Time_consumed : %.3fs" % (time_consumed))
        print("loss:")
        print(loss)
        print("output:")
        print(output)
        print("embedding")
        print(embedding)
        print("a:")
        print(a)
        print("b:")
        print(b)
        print("w0:")
        print(w0)
        print("w:")
        print(w)