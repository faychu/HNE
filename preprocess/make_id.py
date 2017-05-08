# coding:utf-8
from six.moves import cPickle as pickle
from collections import Counter

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

def get_id():
    pc = len(data)
    ac = get_author_count()
    vc = get_venue_count()
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

if __name__ == "__main__":
    with open('data/filter_data_again.pickle', 'rb') as f:
        data = pickle.load(f)
    pid_dic, aid_dic, vid_dic = get_id()
    with open('data/pid.pickle','wb') as f:
        pickle.dump(pid_dic, f, 2)
    with open('data/aid.pickle', 'wb') as f:
        pickle.dump(aid_dic, f, 2)
    with open('data/vid.pickle', 'wb') as f:
        pickle.dump(vid_dic, f, 2)