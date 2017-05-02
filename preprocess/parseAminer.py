# coding=utf-8
import json
import datetime
# import pymongo
# from pymongo import MongoClient
from bson import json_util


def readTextDataFromFile(_file_path):
    with open(_file_path, 'r') as content_file:
        content = content_file.read()
        return content


def readDataFromFileLineByLine(_file_path):
    all_papers = []
    one_paper = {'title': '', 'authors': [], 'year': '', 'venue': '', 'index': '', 'references': [], 'citations': [],
                 'abstract': ''}
    with open(_file_path, 'r') as content_file:
        line_counter = 0
        most_recent_mark = ''
        for one_line in content_file:
            line_counter = line_counter + 1
            one_line = one_line.rstrip('\n').strip(' ').strip(' ')

            # get the most recent mark
            if len(one_line) != 0 and one_line[0] == '#':
                most_recent_mark = one_line[:2]

            if len(one_line) == 0:
                flag = True
                for attr in one_paper:  # check if extra empty row exists
                    flag = flag and (len(one_paper[attr]) == 0)
                if flag:
                    print('========================Error: Wrong empty row!!!!!!!===================' + str(line_counter))
                else:
                    all_papers.append(one_paper)
                    one_paper = {'title': '', 'authors': [], 'year': '', 'venue': '', 'index': '', 'references': [],
                                 'citations': [],
                                 'abstract': ''}
            elif one_line[:2] == '#*':
                one_paper['title'] = one_line[2:]
            elif one_line[:2] == '#@':
                one_paper['authors'] = one_line[2:].split(', ')
            elif one_line[:2] == '#t':
                one_paper['year'] = one_line[2:]
            elif one_line[:2] == '#c':
                one_paper['venue'] = one_line[2:]
            elif one_line[:2] == '#i':
                one_paper['index'] = one_line[6:]
            elif one_line[:2] == '#!':
                one_paper['abstract'] = one_line[2:]
            elif one_line[:2] == '#%':
                if len(one_line[2:]) != 0:
                    one_paper['references'].append(one_line[2:])
            else:
                if most_recent_mark == '#!':
                    one_paper['abstract'] = one_paper['abstract'] + ' ' + one_line
                else:
                    print('========================Error: no match!!!!!!!===================' + str(line_counter))

        print('line_counter: ' + str(line_counter))
        return all_papers


def list2Dict(_data_list):
    data_dict = {}
    for i in range(0, len(_data_list)):
        idx = _data_list[i]['index']
        data_dict[idx] = _data_list[i]

    return data_dict


def updateCitations(_data_list, _data_dict):
    for i in range(0, len(_data_list)):
        for j in range(0, len(_data_list[i]['references'])):
            refer_id = _data_list[i]['references'][j]
            cur_id = _data_list[i]['index']
            _data_dict[refer_id]['citations'].append(cur_id)
    return _data_dict


def getPapersOfResearcher(_data_dict, _researcher):
    paper_list = []
    found = False
    for attr, value in _data_dict.iteritems():
        found = False
        for i in range(0, len(value['authors'])):
            for j in range(0, len(_researcher)):
                if value['authors'][i] == _researcher[j]:
                    paper_list.append(value)
                    found = True
                    break
            if found:
                break
    return paper_list


def saveList2JsonFile(_file_path, _data_list):
    with open(_file_path, 'w') as f:
        f.write(json_util.dumps(_data_list)) #!!!!The correct way to tranform JSON-unserializable data to JSON-serializable data

if __name__ == "__main__":
    # process data and dump into .pickle file
    result = readDataFromFileLineByLine('data/test.txt')
    print(result)
    # dump2pickle(result, 'data/test.pickle')
    # saveList2JsonFile(result, '/data/test.json')