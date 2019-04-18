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


# 输出层采用softmax，它与LSTM进行全连接。对于每一个字符来说，它经过LSTM后的输出
# 大小是1 x L（L为LSTM cell隐层的结点数量），我们上面也分析过输入一个N x M的batch，
# 我们从LSTM层得到的输出为N x M x L，要将这个输出与softmax全连接层建立连接，就需要
# 对LSTM的输出进行重塑，变成(N*M) x L 的一个2D的tensor。softmax层的结点数应该
# 是vocab的大小（我们要计算概率分布）。因此整个LSTM层到softmax层的大小为L x vocab_size。

def build_output(lstm_output, in_size, out_size):
    """
    构建输出层
    :param lstm_output: lstm层的输出结果（是一个三维数组）
    :param in_size: lstm层重塑后的size
    :param out_size: softmax层的size
    :return:
    """
    # 将lstm的输出按照列concate，例如[[1,2,3],[7,8,9]]，
    # tf.concat的结果是[1,2,3,7,8,9]
    seq_output = tf.concat(lstm_output, 1)
    # reshapes
    x = tf.reshape(seq_output, [-1, in_size])

    # 连接LSTM输入到softmax layer
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))
    # 计算logits
    logits = tf.matmul(x, softmax_w) + softmax_b
    # softmax层返回概率分布
    out = tf.nn.softmax(logits, name='predictions')
    return out, logits


def build_loss(logits, targets, lstm_size, num_classes):
    """
    根据logits和targets计算损失
    :param logits: 全连接层输出的结果（没有经过softmax）
    :param targets: 目标字符
    :param lstm_size: lstm cell隐层结点的数量
    :param num_classes:
    :return:
    """
    # 对target进行编码
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    # softmax cross entropy between logits and labels
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    return loss


# RNN会遇到梯度爆炸（gradients exploding）和梯度弥散（gradients disappearing)的问题。
# LSTM解决了梯度弥散的问题，但是gradients仍然可能会爆炸，因此我们采用gradient clippling
# 的方式来防止梯度爆炸。即通过设置一个阈值，当gradients超过这个阈值时，就将它重置为阈值大小，
# 这就保证了梯度不会变得很大。

def build_optimizer(loss, learning_rate, grad_clip):
    """
    构造Optimizer
    :param loss: 损失
    :param learning_rate: 学习率
    :param grad_clip:
    :return:
    """
    # 使用clipping gradients
    tvars = tf.trainable_variables()
    # tf.clip_by_global_norm会返回clip以后的gradients以及global_norm
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    return optimizer


class CharRNN:
    # num_classes等于字典大小
    def __init__(self, num_classes, batch_size=64, num_steps=50, lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False):
        # 如果sampling是True，则采用SGD
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps
        tf.reset_default_graph()
        # 输入层
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)
        # LSTM层
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)
        # 对输入进行one-hot编码
        x_one_hot = tf.one_hot(self.inputs, num_classes)
        # 运行RNN
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state
        # 预测结果
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)
        # Loss和Optimizer(with gradient clipping)
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)


# batch_size: 单个batch中序列的个数
# num_steps: 单个序列中字符数目
# lstm_size: 隐层结点个数
# num_layers: LSTM层个数
# learning_rate: 学习率
# keep_prob: 训练时
# dropout层中保留结点比例

batch_size = 100
num_steps = 100
lstm_size = 512
num_layers = 2
learning_rate = 0.001
keep_prob = 0.5

epochs = 20
# 每n轮进行一次变量保存
save_every_n = 200

model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps, lstm_size=lstm_size, num_layers=num_layers,
                learning_rate=learning_rate)
saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    counter = 0
    for e in range(epochs):
        # Train network
        new_state = sess.run(model.initial_state)
        loss = 0
        for x, y in get_batches(encoded, batch_size, num_steps):
            counter += 1
            start = time.time()
            feed = {
                model.inputs: x,
                model.targets: y,
                model.keep_prob: keep_prob,
                model.initial_state: new_state
            }
            batch_loss, new_state, _ = sess.run([
                model.loss,
                model.final_state,
                model.optimizer
            ], feed_dict=feed)
            end = time.time()
            # control the print lines
            if counter % 500 == 0:
                print(
                    '轮数：{}/{}...'.format(e + 1, epochs),
                    '训练步数：{}...'.format(counter),
                    '训练误差：{:.4f}...'.format(batch_loss),
                    '{:.4f} sec/batch'.format((end - start))
                )
            if counter % save_every_n == 0:
                saver.save(sess, 'checkpoints/i{}_l{}.ckpt'.format(counter, lstm_size))
    saver.save(sess, 'checkpoints/i{}_l{}.ckpt'.format(counter, lstm_size))
