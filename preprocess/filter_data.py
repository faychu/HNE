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
    print('rare authors #: '+ str(len(rare_author)))
    return rare_author

def get_rare_venue(num):
    rare_venue = []
    with open('data/venue_count.pickle', 'rb') as f:
        data = pickle.load(f)
    # 定义出现次数少于num次的为rare author
    for i in data:
        if data[i] <= num:
            rare_venue.append(i)
    print('rare venues #: ' + str(len(rare_venue)))
    return rare_venue

def filter(rare_author, rare_venue):
    with open('data/publications.pickle', 'rb') as f:
        records = pickle.load(f)
    new_records = []
    for record in records:
        if record['venue'] not in rare_venue:
            authors = []
            for author in record['authors']:
                if author not in rare_author:
                    authors.append(author)
            if len(authors) != 0:
                record['authors'] = authors
                new_records.append(record)
            if len(new_records) % 10 == 0:
                print(len(new_records))
    return new_records

if __name__ == "__main__":
    with open('data/author_label_dict.pickle','rb') as f:
        dic = pickle.load(f)
    rare_author = get_rare_author(4)
    count = 0
    rare_author_label = {}
    f = open('data/rare_author_label.txt','w')
    for i in rare_author:
        if i in dic:
            dic.pop(i)
    count = [0,0,0,0]
    for i in dic:
        if dic[i] == '1':
            count[0]+=1
        elif dic[i] == '2':
            count[1]+=1
        elif dic[i] == '3':
            count[2]+=1
        elif dic[i] == '4':
            count[3]+=1
    print(count)
    rare_venue = get_rare_venue(4)
    with open('data/rare_author.pickle', 'wb') as f:
        pickle.dump(rare_author, f, pickle.HIGHEST_PROTOCOL)
    with open('data/rare_venue.pickle', 'wb') as f:
        pickle.dump(rare_venue, f, pickle.HIGHEST_PROTOCOL)
    filter_data = filter(rare_author,rare_venue)
    print(len(filter_data))
    with open('data/filter_data.pickle','wb') as f:
        pickle.dump(filter_data,f,pickle.HIGHEST_PROTOCOL)