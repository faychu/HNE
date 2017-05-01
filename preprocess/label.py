from six.moves import cPickle as pickle
from collections import Counter
from matplotlib import pyplot
import numpy as np
# 123
def get_paper_list():
    paper_list = []
    with open('data/publications.pickle','rb') as f:
        data = pickle.load(f)
    for i in data:
        paper_list.append(i['title'])
    return paper_list

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
    print len(label_papers_list),len(paper_not_in_data)
    return label_papers_list, paper_not_in_data

if __name__ == "__main__":
    label_files = ['Data Mining', 'Database', 'Medical Informatics', 'Theory', 'Visualization']
    paper_list = get_paper_list()
    for label_file in label_files:
        label_papers_list, paper_not_in_data = read_label_data(label_file.join(['data/','.txt']), paper_list)
        with open(label_file.join(['data/','.pickle']), 'wb') as f:
            pickle.dump(label_papers_list, f, pickle.HIGHEST_PROTOCOL)