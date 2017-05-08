# coding:utf-8
from six.moves import cPickle as pickle
from collections import Counter
from matplotlib import pyplot
import numpy as np
#####################
# generate data #
# non_publications.pickle
# filter_records.pickle
# pid、aid、vid（.pickle, .txt）
# ['Data Mining', 'Database', 'Medical Informatics', 'Theory', 'Visualization'](.pickle)
# author_label_out(.pickle, .txt)

def get_paper_counter(data):
    paper_list = []
    one_paper_list = []
    for i in data:
        paper_list.append(i['title'])
    c = Counter(paper_list)
    for i in c:
        if c[i] ==1:
            one_paper_list.append(i)
    return one_paper_list

def get_author_count(data):
    author_list = []
    for i in data:
        author_list.extend(i['authors'])
    c = Counter(author_list)
    print("# A: "+str(len(c)))
    return c

def get_venue_count(data):
    venue_list = []
    for i in data:
        venue_list.append(i['venue'])
    c = Counter(venue_list)
    print("# V: " + str(len(c)))
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

def get_rare_author(ac,num):
    rare_author = []
    # 定义出现次数少于num次的为rare author
    for i in ac:
        if ac[i] <= num:
            rare_author.append(i)
    print('rare authors #: '+ str(len(rare_author)))
    return rare_author

def get_rare_venue(vc,num):
    rare_venue = []
    # 定义出现次数少于num次的为rare author
    for i in vc:
        if vc[i] <= num:
            rare_venue.append(i)
    print('rare venues #: ' + str(len(rare_venue)))
    return rare_venue

def filter_data(records,rare_author, rare_venue):
    rare_author = set(rare_author)
    rare_venue = set(rare_venue)
    filter_records = []
    for record in records:
        if record['venue'] not in rare_venue:
            authors = []
            for author in record['authors']:
                if author not in rare_author:
                    authors.append(author)
            if len(authors) != 0:
                record['authors'] = authors
                filter_records.append(record)
    return filter_records

def get_id(data):
    pc = len(data)
    ac = get_author_count(data)
    vc = get_venue_count(data)
    pid_dic = {}
    aid_dic = {}
    vid_dic = {}
    pid = 0
    aid = pc
    vid = len(ac)+len(data)
    # get paper id
    fp = open('data/pid.txt','w')
    fa = open('data/aid.txt','w')
    fv = open('data/vid.txt','w')
    # frepeat = open('data/repeat.txt','w')
    for record in data:
        pid_dic[record['title']] = pid
        fp.write(record['title'] + '\t' + str(pid) + '\n')
        pid += 1
    # get author id
    for a in ac:
        aid_dic[a] = aid
        fa.write(a+'\t'+str(aid)+'\n')
        aid += 1
    # get venue id
    for v in vc:
        vid_dic[v] = vid
        fv.write(v+'\t'+str(vid)+'\n')
        vid += 1
    return pid_dic, aid_dic, vid_dic

def read_label_data(_file_path, paper_list):
    label_papers_list = []
    paper_not_in_data = []
    with open(_file_path, 'r') as f:
        for line in f:
            label_paper = line.strip().split('\t')[1]
            if label_paper in paper_list:
                label_papers_list.append(label_paper)
            else:
                paper_not_in_data.append(label_paper)
    print(len(label_papers_list), len(paper_not_in_data))
    return label_papers_list, paper_not_in_data

if __name__ == "__main__":
    ###### filter data ##########
    # with open('data/publications.pickle', 'rb') as f:
    #     records = pickle.load(f)
    # new_records = []
    # paperset = set(get_paper_counter(records))
    # for record in records:
    #     if record['title'] in paperset:
    #         new_records.append(record)
    # print(len(new_records))
    # with open('data/non_publications.pickle', 'wb') as f:
    #     pickle.dump(new_records, f, 2)
    # ac = get_author_count(new_records)
    # vc = get_venue_count(new_records)
    #
    # rare_v = get_rare_venue(vc,4)
    # rare_a = get_rare_author(ac,4)
    #
    # filter_records = filter_data(new_records,rare_a,rare_v)
    # with open('data/filter_records.pickle','wb') as f:
    #     pickle.dump(filter_records,f,2)
    #
    # ac = get_author_count(filter_records)
    # vc = get_venue_count(filter_records)
    #
    # rare_v = get_rare_venue(vc, 4)
    # rare_a = get_rare_author(ac, 4)

    ######## test and get id ############
    # with open('data/filter_records.pickle', 'rb') as f:
    #     data = pickle.load(f)
    # print(len(data))
    # print(data[0:10])
    # pid_dic, aid_dic, vid_dic = get_id(data)
    # with open('data/pid.pickle', 'wb') as f:
    #     pickle.dump(pid_dic, f, 2)
    # with open('data/aid.pickle', 'wb') as f:
    #     pickle.dump(aid_dic, f, 2)
    # with open('data/vid.pickle', 'wb') as f:
    #     pickle.dump(vid_dic, f, 2)
    #
    # ######## check label #############
    # # check paper label
    # label_files = ['Data Mining', 'Database', 'Medical Informatics', 'Theory', 'Visualization']
    # for label_file in label_files:
    #     label_papers_list, paper_not_in_data = read_label_data(label_file.join(['data/', '.txt']), pid_dic)
    #     with open(label_file.join(['data/', '.pickle']), 'wb') as f:
    #         pickle.dump(label_papers_list, f, pickle.HIGHEST_PROTOCOL)
    # # check author label
    # author_dict = {}
    # author_label = {}
    # author_label_out = {}
    # fout = open('author_label_out.txt','w')
    # with open('data/author_dict_utf8.txt',"r") as f:
    #     for line in f :
    #         line = line.strip().split('\t')
    #         author_dict[line[0]] = line[1]
    # with open('data/author_label_utf8.txt',"r") as f:
    #     for line in f :
    #         line = line.strip().split('\t')
    #         author_label[line[0]] = line[1]
    # for i in author_label:
    #     if author_dict[i] in aid_dic:
    #         author_label_out[author_dict[i]] = author_label[i]
    #         fout.write(author_dict[i]+'\t'+author_label[i]+'\n')
    # with open('data/author_label_out.pickle','wb') as f:
    #     pickle.dump(author_label_out, f, 2)
    # count = [0, 0, 0, 0]
    # for i in author_label_out:
    #     if author_label_out[i] == '1':
    #         count[0] += 1
    #     elif author_label_out[i] == '2':
    #         count[1] += 1
    #     elif author_label_out[i] == '3':
    #         count[2] += 1
    #     elif author_label_out[i] == '4':
    #         count[3] += 1
    # print(count)

    ############## make data for training ###########
    with open('data/filter_records.pickle', 'rb') as f:
        data = pickle.load(f)
    with open('data/pid.pickle', 'rb') as f:
        pid_dic = pickle.load(f)
    with open('data/aid.pickle', 'rb') as f:
        aid_dic = pickle.load(f)
    with open('data/vid.pickle', 'rb') as f:
        vid_dic = pickle.load(f)
    records = []
    for i in data:
        record = {}
        record['p'] = int(pid_dic[i['title']])
        record['a'] = [int(aid_dic[j]) for j in i['authors']]
        record['v'] = int(vid_dic[i['venue']])
        records.append(record)
    print(records[0:10])
    with open('data/final_records.pickle','wb') as f:
        pickle.dump(records, f, 2)
