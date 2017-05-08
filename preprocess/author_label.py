# coding:utf-8
__author__ = 'fay'
from six.moves import cPickle as pickle
from collections import Counter
from matplotlib import pyplot
import numpy as np
import codecs

def get_author_list():
    author_list = []
    for i in data:
        author_list.extend(i['authors'])
    return author_list


if __name__ == "__main__":
    # with open('data/publications.pickle','rb') as f:
    #     data = pickle.load(f)
    #
    # author_list = get_author_list()
    # print(author_list[0:10])

    # with open('data/authorlist.pickle', "rb") as f:
    #     author_list = pickle.load(f)
    # print(author_list[0:10])
    # author_dict = {}
    # author_label = {}
    # author_label_out = {}
    # with open('data/author_dict_utf8.txt',"r") as f:
    #     for line in f :
    #         line = line.strip().split('\t')
    #         author_dict[line[0]] = line[1]
    # with open('data/author_label_utf8.txt',"r") as f:
    #     for line in f :
    #         line = line.strip().split('\t')
    #         author_label[line[0]] = line[1]
    #
    # for i in author_label:
    #     if author_dict[i] in author_list:
    #         author_label_out[author_dict[i]] = author_label[i]
    #
    # with open('data/author_label_dict.pickle','wb') as f:
    #     pickle.dump(author_label_out, f, 2)
    #
    # with open('data/author_label_dict.pickle','rb') as f:
    #     author_label_out = pickle.load(f)
    #
    # count = 0
    # for i in author_list[0:10]:
    #     if i in author_label_out:
    #         count+=1
    #     else:
    #         print(i)
    # print(count)
    # print(len(author_label_out))

    with open('data/author_label_dict.pickle','rb') as f:
        dic = pickle.load(f)
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