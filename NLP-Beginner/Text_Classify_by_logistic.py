import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from itertools import chain
from collections import Counter


def get_vocabulary(words):
    '''
    @description: 从语料中构建词汇表
    @param {type} 
    @return: 词汇表(list)
    '''
    word_list = [set(x.split()) for x in words]
    return list(chain(*word_list))


def bag_of_word(words, vocabulary):
    '''
    @description: 将文本数据转换为词袋模型向量
    @param {type}
    @return: 词袋模型向量(narray)
    '''
    word_list = [x.split() for x in words]
    bag = np.zeros((len(word_list), len(vocabulary)))
    for index, line in enumerate(word_list):
        req_dict = Counter(line)
        for word in req_dict:
            word_index = vocabulary.index(word)
            bag[index][word_index] = req_dict[word]
    return bag


def vocabulary_of_word(words, vocabulary, max_length):
    '''
    @description: 将文本数据转换为词汇表模型向量
    @param {type} 
    @return: 词汇表向量(narray)
    '''
    word_list = [x.split() for x in words]
    vocab = np.zeros((len(word_list), max_length))
    for index, line in enumerate(word_list):
        for word_index, word in enumerate(line[:max_length]):
            vocab[index][word_index] = vocabulary.index(word)
    return vocab


def N_gram():
    pass


def logistic():
    pass


def softmax():
    pass


def loss_fuction():
    pass


def gradient_descent():
    pass


# 讀取數據文本
data = pd.read_csv('data/train.tsv', usecols=[3])
data_x_list = np.array(data).tolist()
# 讀取數據標簽
y = pd.read_csv('data/train.tsv', usecols=[4])
data_y_list = np.array(y).tolist()
