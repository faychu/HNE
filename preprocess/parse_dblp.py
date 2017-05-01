# coding=utf-8
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#
# File Name : parse_dblp.py
#
# Purpose : Parse the Aminer data
# https://cn.aminer.org/citation
#
# #* --- paperTitle
# #@ --- Authors
# #t ---- Year
# #c  --- publication venue
# #index 00---- index id of this paper
# #% ---- the id of references of this paper (there are multiple lines, with each indicating a reference)
# #! --- Abstract
#
# Creation Date : 27-04-2017
#
# Last Modified : Thu 27 Apr 2017
#
# Created By : Yunfei Chu (yfchu@bupt.edu.cn)
#
# _._._._._._._._._._._._._._._._._._._._._.
import json
import datetime
from six.moves import cPickle as pickle
from bson import json_util

def readDataFromFileLineByLine(_file_path):
    all_papers = []
    incomplete_paperID = []

    one_paper = {'title': '', 'authors': [], 'year': '', 'venue': '', 'index': '', 'references': [], 'citations': [],
                 'abstract': ''}
    with open(_file_path, 'r') as content_file:
        line_counter = 0

        for one_line in content_file:
            line_counter += 1
            one_line = one_line.rstrip('\n').strip(' ').strip(' ')
            if len(one_line) == 0:  # black line
                if len(one_paper['authors']) != 0 and one_paper['venue'] != '':
                    all_papers.append(one_paper)
                else:
                    incomplete_paperID.append(one_paper['title'])
                one_paper = {'title': '', 'authors': [], 'year': '', 'venue': '', 'index': '', 'references': [],
                            'citations': [],
                            'abstract': ''}
            elif one_line[:2] == '#*':
                one_paper['title'] = one_line[2:]
            elif one_line[:2] == '#@':
                if one_line[2:] != '':
                    one_paper['authors'] = one_line[2:].split(',')
            elif one_line[:2] == '#t':
                one_paper['year'] = one_line[2:]
            elif one_line[:2] == '#c':
                one_paper['venue'] = one_line[2:]
            elif one_line[:2] == '#i':
                one_paper['index'] = one_line[6:]
            elif one_line[:2] == '#!':
                one_paper['abstract'] = one_line[2:]
            elif one_line[:2] == '#%':
                if one_line[2:] != '':
                    one_paper['references'].append(one_line[2:])
        print 'line_counter: ' + str(line_counter)
        print 'paper_counter:'+str(paperID)
        return all_papers,incomplete_paperID

def dump2pickle(data, _file_path):
    with open(_file_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def dump2txt(data,_file_path):
    with open(_file_path, 'w') as f:
        for i in data:
            f.writelines(i+'\n')

def saveList2JsonFile(_data_list,_file_path ):
    with open(_file_path, 'w') as f:
        f.write(json_util.dumps(_data_list))

if __name__ == "__main__":
    # process data and dump into .pickle and .json file
    result, incomplete_paperID = readDataFromFileLineByLine('data/publications.txt')
    print len(result)
    # dump2pickle(result, 'data/publications.pickle')
    # saveList2JsonFile(result, 'data/publications.json')
    # dump2txt(incomplete_paperID, 'data/incomplete.txt')
