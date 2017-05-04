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
from utils.utils import *
import time
import scipy.io as sio

if __name__ == "__main__":
    config = Config()

    records_data = Records(config.file_path)
    config.struct['input_dim'] = records_data.num_nodes

    model = NMFM(config)
    model.do_variables_init()

    last_loss = np.inf  # 无限大的正数
    converge_count = 0
    time_consumed = 0
    epochs = 0
    while (True):
        mini_batch = records_data.sample(config.batch_size)
        st_time = time.time()
        model.fit(mini_batch)
        time_consumed += time.time() - st_time

        if records_data.is_epoch_end:
            epochs += 1
            loss = 0
            embedding = None
            while (True):
                mini_batch = records_data.sample(config.batch_size, do_shuffle = False)
                loss += model.get_loss(mini_batch)
                if embedding is None:
                    embedding = model.get_embedding(mini_batch)
                else:
                    embedding = np.vstack((embedding, model.get_embedding(mini_batch)))

                if records_data.is_epoch_end:
                    break

            print("Epoch : %d Loss : %.3f, Train time_consumed : %.3fs" % (epochs, loss, time_consumed))

            if (loss > last_loss):
                converge_count += 1
                if converge_count > 10:
                    print "model converge terminating"
                    check_link_reconstruction(embedding, graph_data, [1000,3000,5000,7000,9000,10000])
                    break
            if epochs > config.epochs_limit:
                print "exceed epochs limit terminating"
                break
            last_loss = loss


    sio.savemat(config.embedding_filename + '_embedding.mat',{'embedding':embedding})
