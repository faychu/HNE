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
import numpy as np
from utils.utils import *
from six.moves import cPickle as pickle

class Records(object):
    def __init__(self, pickle_path):
        self.num_nodes = 2085231
        self.st = 0
        self.is_epoch_end = True
        with open(pickle_path, "rb") as f:
            self.records = pickle.load(f)
        self.N = len(self.records)
        self.X_sp_indices = []
        self.X_sp_val = []
        self.X_sp_shape = []
        for i in self.records:
            index = []
            value = []
            index.append(i['p'])
            value.append(1.0)
            index.extend(i['a'])
            value.extend([1.0/len(i['a'])]*len(i['a']))
            index.append(i['v'])
            value.append(1.0)
            self.X_sp_indices.append(index)
            self.X_sp_val.append(value)
        self.__order = np.arange(self.N)
        print("Get Data Done!")
        # with open('indices.pickle','wb') as f:
        #     pickle.dump(self.X_sp_indices, f, 2)
        # with open('ids_val','wb') as f:
        #     pickle.dump(self.X_sp_ids_val, f, 2)
        # print("Dump Data Done!")

    def sample(self, batch_size, do_shuffle=True):  # Finished!
        if self.is_epoch_end:
            if do_shuffle:
                np.random.shuffle(self.__order)
            else:
                self.__order = np.sort(self.__order)
            self.st = 0
            self.is_epoch_end = False

        mini_batch = Dotdict()
        en = min(self.N, self.st + batch_size)
        index = self.__order[self.st:en]
        # mini_batch.batch_size = len(index)
        if en == self.N:
            index=np.hstack((index, self.__order[0:batch_size-len(index)]))
            assert len(index) == batch_size
        mini_batch.records = [self.X_sp_indices[i] for i in index]
        # mini_batch.X_sp_indices = [[i,j] for i in range(len(index)) for j in self.X_sp_indices[index[i]]]
        mini_batch.X_sp_indices = []
        mini_batch.X_sp_val = []
        mini_batch.X_val_ids = []
        # mini_batch.X_sp_ids_val = [i for j in range(len(index)) for i in self.X_sp_ids_val[index[j]]]
        # mini_batch.X_sp_indices = [[i,j] for i in range(len(index)) for j in mini_batch.records[i]]
        c = 0
        for i in range(len(index)):
            for j in mini_batch.records[i]:
                mini_batch.X_sp_indices.append([i,j])
                mini_batch.X_val_ids.append([i,c])
                c +=1
            mini_batch.X_sp_val.extend([j for j in self.X_sp_val[index[i]]])
        mini_batch.X_indices = np.split(np.array(mini_batch.X_sp_indices), 2, axis=1)[1].reshape([-1])
        # mini_batch.X_sp_ids_val = [self.X_sp_ids_val[i] for i in index]
        mini_batch.X_sp_shape = [batch_size, self.num_nodes]
        mini_batch.X_val_shape = [batch_size, c]
        if (en == self.N):
            en = 0
            self.is_epoch_end = True
        self.st = en
        return mini_batch

