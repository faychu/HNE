# coding:utf-8
from six.moves import cPickle as pickle
from collections import Counter
from matplotlib import pyplot
import numpy as np
def get_paper_counter(data):
    paper_list = []
    for i in data:
        paper_list.append(i['title'])
    return Counter(paper_list)



if __name__ == "__main__":
    with open('data/filter_data.pickle', 'rb') as f:
        records = pickle.load(f)
    with open('data/paper_norepeat.pickle','rb') as f:
        data =set(pickle.load(f))
    new_records = []
    for record in records:
        if record['title'] in data:
            new_records.append(record)
    print(len(new_records))
    with open('data/filter_data_again.pickle','wb') as f:
        pickle.dump(new_records, f, 2)

    # c = get_paper_counter(data)
    # paperl=[]
    # for i in c:
    #     if c[i] ==1:
    #         paperl.append(i)
    # with open('data/paper_norepeat.pickle','wb') as f:
    #     pickle.dump(paperl,f,2)
    # print len(paperl)



