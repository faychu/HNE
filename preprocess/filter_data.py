# coding=utf-8
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#
# File Name : filter_data.py
#
# Purpose : filter rare author
# https://cn.aminer.org/citation
#
# Creation Date : 29-04-2017
#
# Last Modified : Thu 29 Apr 2017
#
# Created By : Yunfei Chu (yfchu@bupt.edu.cn)
#
# _._._._._._._._._._._._._._._._._._._._._.

from six.moves import cPickle as pickle
from collections import Counter
from matplotlib import pyplot
import numpy as np

def get_rare_author(num):
    rare_author = []
    with open('data/author_count.pickle', 'rb') as f:
        data = pickle.load(f)
    # 定义出现次数少于num次的为rare author
    for i in data:
        if data[i] <= num:
            rare_author.append(i)
    print 'rare authors #: '+ str(len(rare_author))
    return rare_author

def get_rare_venue(num):
    rare_venue = []
    with open('data/venue_count.pickle', 'rb') as f:
        data = pickle.load(f)
    # 定义出现次数少于num次的为rare author
    for i in data:
        if data[i] <= num:
            rare_venue.append(i)
    print 'rare venues #: ' + str(len(rare_venue))
    return rare_venue

def filter(rare_author, rare_venue):
    with open('data/publications.pickle', 'rb') as f:
        records = pickle.load(f)
    for record in records:
        if record['venue'] in rare_venue:
            records.remove(record)
        for author in record['authors']:
            if author in rare_author:
                record['authors'].remove(author)
                if len(record['authors']) == 0:
                    records.remove(record)
    return records

if __name__ == "__main__":
    rare_author = get_rare_author(5)
    rare_venue = get_rare_venue(5)
    with open('data/rare_author.pickle', 'wb') as f:
        pickle.dump(rare_author, f, pickle.HIGHEST_PROTOCOL)
    with open('data/rare_venue.pickle', 'wb') as f:
        pickle.dump(rare_venue, f, pickle.HIGHEST_PROTOCOL)