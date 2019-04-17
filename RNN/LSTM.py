#!/usr/bin/env python
# encoding: utf-8
"""
@author: dstch
@license: (C) Copyright 2013-2019, Regulus Tech.
@contact: dstch@163.com
@file: LSTM.py
@time: 2019/4/17 22:24
@desc: https://zhuanlan.zhihu.com/p/27087310
       https://github.com/NELSONZHAO/zhihu/tree/master/anna_lstm

       TensorFlow构建LSTM模型进行英文字符级文本生成

"""

import time
import numpy as np
import tensorflow as tf

# 加载数据
with open('data/anna.txt', 'r') as f:
    text = f.read()

# 构建字符集合
vocab = set(text)
# 字符-数字映射字典
vocab_to_int = {c: i for i, c in enumerate(vocab)}
# 数字-字符映射字典
int_to_vocab = dict(enumerate(vocab))
# 对文本进行转码
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)


# mini-batch
def get_batches(arr, n_seqs, n_steps):
    """
    对已有的数组进行mini-batch分割
    :param arr: 待分割的数组
    :param n_seqs: 一个batch中的序列个数
    :param n_steps: 单个序列长度
    :return:
    """
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)

    # 对不能整除的部分进行舍弃
    arr = arr[: batch_size * n_batches]
    # 重塑
    arr = arr.reshape((n_seqs, -1))
    for n in range(0, arr.shape[1], n_steps):
        x = arr[:n:n + n_steps]
        # 注意targets相比于x会向后错位一个字符
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], y[:, 0]
        yield x, y


# 模型构建

def build_inputs(num_seqs, num_steps):
    """
    构建输入层
    :param num_seqs: 每个batch中的序列个数
    :param num_steps: 每个序列包含的字符数
    :return:
    """
    inputs = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='inputs')
    targets = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='targets')
    # 加入keep_prob，控制dropout的保留结点数
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, keep_prob


def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    """
    构建LSTM层
    :param lstm_size: lstm cell中隐层结点数目
    :param num_layers: lstm层的数目
    :param batch_size: num_seqs x num_steps
    :param keep_prob:
    :return:
    """
    # 构建一个基本LSTM单元
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    # 添加dropout
    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    # 叠堆多个LSTM单元
    cell = tf.nn.rnn_cell.MultiRNNCell([drop for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)

    return cell, initial_state
