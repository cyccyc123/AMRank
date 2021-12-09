from random import *
import numpy as np

def feature_normal(dic, feature_dim):
    mean = np.zeros(feature_dim)
    doc_num = 0
    for qid in dic:
        pairs = dic[qid]
        for pair in pairs:
            mean += pair[0]
            doc_num += 1
    mean /= doc_num
    sigma = np.zeros(feature_dim)
    for qid in dic:
        pairs = dic[qid]
        for pair in pairs:
            sigma += (pair[0] - mean) * (pair[0] - mean)
    sigma /= doc_num
    sigma = np.sqrt(sigma)
    for i in range(len(sigma)):
        if sigma[i] == 0:
            sigma[i] = 1
    # calcu normal data
    for qid in dic:
        pairs = dic[qid]
        for pair in pairs:
            pair[0] = (pair[0] - mean) / sigma
    return dic


def get_batch(path, feature_dim):
    '''
    function: get all candidate document of the query
    df: all data
    batch_size: Maximum length of candidate documents for all pueries
    '''
    input_num = 2
    dic = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            feat, comment = line.split(' #')
            feat = feat.split()
            yi = feat[0]
            qid = feat[1].split(':')[-1]
            xi = feat[2:]
            xi = [f.split(':')[-1] for f in xi]
            xi = np.array(xi, dtype=np.float32)
            qid = int(qid)
            yi = int(yi)
            if qid in dic:
                dic[qid].append([xi, yi])
            else:
                dic.update({qid: []})
                dic[qid].append([xi, yi])
    dic = feature_normal(dic, feature_dim)

    for qid in dic:
        pairs = dic[qid]
        shuffle(pairs)
        qid_batchlen = len(pairs)
        yield [np.array([pair[j] for pair in pairs]) for j in range(input_num)] + [qid_batchlen] + [np.array(qid)]


def get_batch_with_test(path, feature_dim):
    '''
    function: get all candidate document of the query
    df: all data
    batch_size: Maximum length of candidate documents for all pueries
    '''
    input_num = 2
    dic = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            feat, comment = line.split(' #')
            feat = feat.split()
            yi = feat[0]
            qid = feat[1].split(':')[-1]
            xi = feat[2:]
            xi = [f.split(':')[-1] for f in xi]
            xi = np.array(xi, dtype=np.float32)
            qid = int(qid)
            yi = int(yi)
            if qid in dic:
                dic[qid].append([xi, yi])
            else:
                dic.update({qid: []})
                dic[qid].append([xi, yi])
    dic = feature_normal(dic, feature_dim)

    for qid in dic:
        pairs = dic[qid]
        shuffle(pairs)
        qid_batchlen = len(pairs)
        yield [np.array([pair[j] for pair in pairs]) for j in range(input_num)] + [qid_batchlen] + [np.array(qid)]
