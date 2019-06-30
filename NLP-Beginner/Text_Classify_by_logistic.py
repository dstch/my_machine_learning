import numpy as np
import pandas as pd
from itertools import chain
from collections import Counter
from math import exp
import random


def load_data(data_path='data/train.tsv'):
    # 讀取數據文本
    data = pd.read_csv(data_path, usecols=[2])
    data_x_list = np.array(data).tolist()
    # 讀取數據標簽
    y = pd.read_csv(data_path, usecols=[3])
    data_y_list = np.array(y).tolist()
    return data_x_list, data_y_list


def split_dataset(df):
    pass


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


def sigmoid(inX):
    # 每個特徵乘以權重，然後把所有的結果相加
    # 將這個總和帶入sigmoid函數，得到一個範圍
    # 在0~1之間的數值。最後，大於0.5歸入1類，
    # 小於0.5的歸入0類。
    # inX=w0x0+w1x1+...+wnxn
    return 1.0/(1+exp(-inX))


def N_gram():
    pass


def logistic():
    pass


def softmax():
    # softmax regression是logistic regression的多分类拓展
    pass


def loss_fuction():
    pass

# region 求导代码，暂时无用


def regularize(xMat):
    inMat = xMat. copy()
    inMeans = np. mean(inMat, axis=0)
    invar = np. std(inMat, axis=0)
    inMat = (inMat-inMeans)/invar
    return inMat


def BGD_LR(dataset, alpha=0.001, maxcycles=500):
    # 参考 http://sofasofa.io/tutorials/python_gradient_descent/
    xMat = np. mat(dataset)
    yMat = np. mat(dataset).T
    xMat = regularize(xMat)
    m, n = xMat.shape
    weights = np. zeros((n, 1))
    for i in range(maxcycles):
        grad = xMat.T*(xMat * weights-yMat)/m
        weights = weights - alpha * grad
    return weights
# endregion


def gradient_descent(dataMatIn, classLabels):
    '''
    @description: 从机器学习实战梯度上升优化算法抄来的
    @param {type}
    @return:
    '''
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001  # 梯度下降步長
    maxCycles = 500  # 迭代次數
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat-h)
        # 这里更新用矩阵乘以差值是因为采用了最小二乘法损失函数，
        # 求导后正好等于矩阵乘以差值
        # https://www.jianshu.com/p/ec3a47903768
        weights = weights+alpha*dataMatrix.transpose()*error
    return weights


def stocGradAscent0(dataMatrix, classLabels):
    # 随机梯度上升算法
    # 每次循环更新一个参数，此处无迭代
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i]-h
        weights = weights+alpha*error*dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    # 改进随机梯度上升算法
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    dataIndex = range(m)
    for j in range(numIter):
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01  # 学习率衰减
            randIndex = int(random.uniform(0, len(dataIndex)))  # 随机选取更新
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex]-h
            weights = weights+alpha*error*dataMatrix[randIndex]
            del(dataMatrix[randIndex])
    return weights
