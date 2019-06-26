#!/usr/bin/env python
# encoding: utf-8
"""
@author: dstch
@license: (C) Copyright 2013-2019, Regulus Tech.
@contact: dstch@163.com
@file: multi-lstm_tensorflow.py
@time: 2019/6/26 21:25
@desc: https://blog.csdn.net/weixin_43568160/article/details/85987572

叠堆多层bi-lstm模型
"""
import tensorflow as tf

# 方法1
n_hidden_units = 50  # 隐藏层神经元数目
num_layers = 3  # 双向lstm神经网络的层数
n_steps = 15
n_inputs = 32
X = tf.placeholder([None, n_steps, n_inputs], dtype=tf.float32)


def lstm_cell_fw():
    # 前向的lstm网络
    return tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)


def lstm_cell_bw():
    # 反向的lstm网络
    return tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)  # 堆叠网络


stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell_fw() for _ in range(num_layers)])
stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell_bw() for _ in range(num_layers)])  # 输出
outputs, _ = tf.nn.bidirectional_dynamic_rnn(stacked_lstm_fw, stacked_lstm_bw, X, dtyep=tf.float32, time_major=False)

# 方法2：
# 单独的定义每一层中的网络，同样还是以3层双向lstm为例，假设每一层双向lstm中隐层的单元状态为50：`
hidden_units = [50, 50, 50]
num_layers = 3
n_steps = 15
n_inputs = 32
X = tf.placeholder([None, n_steps, n_inputs], dtype=tf.float32)
# 前向网络
single_cell_fw = [tf.nn.rnn_cell.LSTMCell(size) for size in hidden_units]
cell_fw = tf.nn.rnn_cell.MultiRNNCell(single_cell_fw)
# 反向网络
single_cell_bw = [tf.nn.rnn_cell.LSTMCell(size) for size in hidden_units]
cell_bw = tf.nn.rnn_cell.MultiRNNCell(single_cell_bw)
outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, X, dtype=tf.float32, time_major=False)

# 方法3：
n_hidden_units = 50  # 隐藏层神经元数目
num_layers = 3  # 双向lstm神经网络的层数
n_steps = 15
n_inputs = 32
X = tf.placeholder([None, n_steps, n_inputs], dtype=tf.float32)
# 定义前向网络
lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
lstm_cell_fw1 = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
# 定义后向网络
lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
lstm_cell_bw1 = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
fw_stack_rnn = [lstm_cell_fw1]
bw_stack_rnn = [lstm_cell_bw1]
for i in range(num_layers):
    fw_stack_rnn.append(lstm_cell_fw)
    bw_stack_rnn.append(lstm_cell_bw)
# 输出
outputs, _ = tf.nn.bidirectional_dyanmic_rnn(fw_stack_rnn, bw_stack_rnn, X, dtype=tf.float32, time_major=False)

# 注意一点：在自己使用cell_fw=tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw() * num_layers,state_is_tuple=True)的时候，
# 编译器总是报cell_fw不是一个instance RNNCell的错误，要求输入的得是instance RNNCell，对于此种情况,我使用了上述的3中堆叠方式，
# 网络都搭建成功了！
