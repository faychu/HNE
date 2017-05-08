# coding=utf-8
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#
# File Name : count_data.py
#
# Purpose : get data id, and filter rare author
# https://cn.aminer.org/citation
#
# Creation Date : 28-04-2017
#
# Last Modified : Thu 28 Apr 2017
#
# Created By : Yunfei Chu (yfchu@bupt.edu.cn)
#
# _._._._._._._._._._._._._._._._._._._._._.

from six.moves import cPickle as pickle
from collections import Counter
from matplotlib import pyplot
import numpy as np

def get_author_count():
    author_list = []
    for i in data:
        author_list.extend(i['authors'])
    c = Counter(author_list)
    print(len(c))
    return c

def get_venue_count():
    venue_list = []
    for i in data:
        venue_list.append(i['venue'])
    c = Counter(venue_list)
    print(len(c))
    return c


def pdf_cdf_plot(counter):
    total = float(sum(counter.values()))
    for key in counter:
        counter[key] /= total
    Y = zip(*sorted(counter.items(), key=lambda a: a[1], reverse=True))[1]
    X = np.arange(len(counter))
    CY = np.cumsum(Y)
    pyplot.plot(X, Y,label='pdf', linewidth=3)
    pyplot.plot(X, CY, label='cdf', linewidth=3)
    pyplot.xlabel('Total')
    pyplot.ylabel('probability')
    pyplot.show()


if __name__ == "__main__":
    with open('data/publications.pickle','rb') as f:
        data = pickle.load(f)

    vc = get_venue_count()
    with open('data/venue_count.pickle','wb') as f:
        pickle.dump(vc, f, pickle.HIGHEST_PROTOCOL)
    print(len(vc))
    # pdf_cdf_plot(vc)
    ac = get_author_count()
    with open('data/author_count.pickle', 'wb') as f:
        pickle.dump(ac, f, pickle.HIGHEST_PROTOCOL)
    print(len(ac))
    # pdf_cdf_plot(ac)




