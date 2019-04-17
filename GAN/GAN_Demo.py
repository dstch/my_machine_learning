#!/usr/bin/env python
# encoding: utf-8
"""
@author: dstch
@license: (C) Copyright 2013-2019, Regulus Tech.
@contact: dstch@163.com
@file: GAN_Demo.py
@time: 2019/4/17 17:45
@desc: https://zhuanlan.zhihu.com/p/43047326
       https://github.com/aadilh/blogs/tree/new/basic-gans/basic-gans

       a simple GAN demo

"""

# build data
import numpy as np


def get_y(x):
    return 10 + x * x


def sample_data(n=10000, scale=100):
    data = []
    x = scale * (np.random.random_sample((n,)) - 0.5)
    for i in range(n):
        yi = get_y(x[i])
        data.append(x[i], yi)
    return np.array(data)


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


# 生成器网络实现  generator net
import tensorflow as tf


def generator(Z, hsize=[16, 16], reuse=False):
    """
    该函数以 placeholder 为随机样本（Z），数组 hsize 为 2 个隐藏层中的神经元数量，
    变量 reuse 则用于重新使用同样的网络层。使用这些输入，函数会创建一个具有 2 个
    隐藏层和给定数量节点的全连接神经网络。函数的输出为一个 2 维向量，对应我们试着
    去学习的真实数据集的维度。对上面的函数也很容易修改，添加更多隐藏层、不同类型的
    层、不同的激活和输出映射等。
    :param Z:
    :param hsize:
    :param reuse:
    :return:
    """
    with tf.variable_scope("GAN/Generator", reuse=reuse):
        # tf.layers.dense(inputs,units,activation):全连接层，units:输出的维度，改变inputs最后一维
        h1 = tf.layers.dense(Z, hsize[0], activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, hsize[1], activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h2, 2)
    return out


# 鉴别器网络实现
def discriminator(X, hsize=[16, 16], reuse=False):
    """
    该函数会将输入 placeholder 认作来自真实数据集向量空间的样本，这些样本可能是真实
    样本，也可能是生成器网络所生成的样本。和上面的生成器网络一样，它也会使用 hsize 和
     reuse 为输入。我们在鉴别器中使用 3 个隐藏层，前两个层的大小和输入中一致，将第三
     个隐藏层的大小修改为 2，这样我们就能在 2 维平面上查看转换后的特征空间，在后面部
     分会讲到。该函数的输出是给定 X 和最后一层的输出（鉴别器从 X 中学习的特征转换）
     的 logit 预测（logit 就是神经网络模型中的 W * X矩阵）。该 logit 函数和 S 型函数
     正好相反，后者常用于表示几率（变量为 1 的概率和为 0 的概率之间的比率）的对数。
    :param X:
    :param hsize:
    :param reuse:
    :return:
    """
    with tf.variable_scope("GAN/Discriminator", reuse=reuse):
        h1 = tf.layers.dense(X, hsize[0], activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, hsize[1], activation=tf.nn.leaky_relu)
        h3 = tf.layers.dense(h2, 2)
        out = tf.layers.dense(h3, 1)
    return out, h3


X = tf.placeholder(tf.float32, [None, 2])  # 真实样本
Z = tf.placeholder(tf.float32, [None, 2])  # 随机噪声样本

# 从生成器中生成样本以及向鉴别器中输入真实和生成的样本
G_sample = generator(Z)
r_logits, r_rep = discriminator(X)
f_logits, g_rep = discriminator(G_sample, reuse=True)

# 生成器和鉴别器的损失函数定义如下
disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(
    r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.ones_like(f_logits)))
# 这些损失是基于 sigmoid 交叉熵的损失，其使用我们前面定义的方程。这种损失函数在离散分类问题中很常见，它将
# logit（由我们的鉴别器网络给定）作为输入，为每个样本预测正确的标签，然后会计算每个样本的误差。

# 用上面所述的损失函数和生成器及鉴别器网络函数中定义的网络层范围，定义这两个网络的优化器。
# 在两个神经网络中我们使用 RMSProp 优化器，学习率设为 0.001，范围则为我们只为给定网络所获取的权重或变量。
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator")

gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss, var_list=gen_vars)  # G Train step
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss, var_list=disc_vars)  # D Train step

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

batch_size = 32
for i in range(100001):
    X_batch = sample_data(n=batch_size)
    Z_batch = sample_Z(batch_size, 2)
    _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})
    _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})

    print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f" % (i, dloss, gloss))
