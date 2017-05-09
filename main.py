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
# from utils.utils import *
import time
import scipy.io as sio

if __name__ == "__main__":
    config = Config()

    records_data = Records(config.file_path)
    print(records_data.records[0:10])
    print(records_data.X_sp_indices[0:10])
    print(records_data.X_sp_val[0:10])
    config.struct['input_dim'] = records_data.num_nodes
    for i in range(10):
        mini_batch = records_data.sample(config.batch_size,do_shuffle=True)
        print(mini_batch.records)
        print(mini_batch.X_sp_indices)
        print(mini_batch.X_indices)
        print(mini_batch.X_sp_val)
        print(mini_batch.X_val_ids)
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
    i = 0
    while (True):
        mini_batch = records_data.sample(config.batch_size)
        i += 1
        st_time = time.time()
        model.fit(mini_batch)
        time_consumed += time.time() - st_time
        loss = model.get_loss(mini_batch)
        if i%100 ==0:
            print("i: %d Epoch : %d Loss : %.3f, Train time_consumed : %.3fs" % (i,epochs, loss, time_consumed))

        if records_data.is_epoch_end:
            epochs += 1
            loss = 0
            embedding = None
            # while (True):
            for i in range(2):
                mini_batch = records_data.sample(config.batch_size, do_shuffle = False)
                loss += model.get_loss(mini_batch)
                if embedding is None:  # todo!
                    embedding = model.get_embedding(mini_batch)
                else:
                    embedding = np.vstack((embedding, model.get_embedding(mini_batch)[:mini_batch.batch_size]))
                if records_data.is_epoch_end:
                    break
                print("Epoch : %d Loss : %.3f, Train time_consumed : %.3fs" % (epochs, loss, time_consumed))

            print("Epoch : %d Loss : %.3f, Train time_consumed : %.3fs" % (epochs, loss, time_consumed))

            if (loss > last_loss):
                converge_count += 1
                if converge_count > 10:
                    print("model converge terminating")
                    print(converge_count)
                    # check_link_reconstruction(embedding, graph_data, [1000,3000,5000,7000,9000,10000])
                    break
            if epochs > config.epochs_limit:
                print("exceed epochs limit terminating")
                break
            last_loss = loss


    sio.savemat(config.embedding_filename + '_embedding.mat',{'embedding':embedding})
