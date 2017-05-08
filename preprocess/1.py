# coding=utf-8
from six.moves import cPickle as pickle
from collections import Counter
from matplotlib import pyplot
import numpy as np
import sys


with open("data/author_count.pickle",'rb') as f:
    authors = pickle.load(f)
print(authors)