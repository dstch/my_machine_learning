#!/usr/bin/env python
# encoding: utf-8
"""
@author: dstch
@license: (C) Copyright 2013-2019, Regulus Tech.
@contact: dstch@163.com
@file: clr_tensorflow.py
@time: 2019/5/17 0:24
@desc: Cyclical Learning Rate in Tensorflow
https://blog.csdn.net/akadiao/article/details/79560731
"""
import matplotlib.pyplot as plt
import tensorflow as tf

"""
TensorFlow中实现的学习率衰减方法：

    tf.train.piecewise_constant　分段常数衰减
    tf.train.inverse_time_decay　反时限衰减
    tf.train.polynomial_decay　多项式衰减
    tf.train.exponential_decay　指数衰减
    tf.train.natural_exp_decay　自然指数衰减
    tf.train.cosine_decay　余弦衰减
    tf.train.linear_cosine_decay　线性余弦衰减
    tf.train.noisy_linear_cosine_decay　噪声线性余弦衰减
    函数返回衰减的学习率．

"""


global_step = tf.Variable(0, name='global_step', trainable=False)
boundaries = [10, 20, 30]
learing_rates = [0.1, 0.07, 0.025, 0.0125]
y = []
N = 40
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for global_step in range(N):
        # -------------------------------- clr --------------------------------------------------
        learing_rate = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=learing_rates)
        lr = sess.run([learing_rate])
        y.append(lr[0])
x = range(N)
plt.plot(x, y, 'r-', linewidth=2)
plt.title('piecewise_constant')
plt.show()
