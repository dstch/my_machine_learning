#!/usr/bin/env python
# encoding: utf-8
"""
@author: dstch
@license: (C) Copyright 2013-2019, Regulus Tech.
@contact: dstch@163.com
@file: keras_capsule_demo.py
@time: 2019/5/17 16:28
@desc: Capsule Net by Keras
https://www.jianshu.com/p/271d5f1f0e25
"""
import keras.backend as K
import numpy as np
from keras.layers import Layer, Input, Dropout, Embedding, Flatten, Dense, Bidirectional, LSTM, Activation
from keras.models import Model
from keras_preprocessing.sequence import pad_sequences
from keras.utils import np_utils


# ---------------- 数据载入 -----------------------
def read_data(data_path):
    senlist = []
    labellist = []
    with open(data_path, "r", encoding='gb2312', errors='ignore') as f:
        for data in f.readlines():
            data = data.strip()
            sen = data.split("\t")[2]
            label = data.split("\t")[3]
            if sen != "" and (label == "0" or label == "1" or label == "2"):
                senlist.append(sen)
                labellist.append(label)
            else:
                pass
    assert (len(senlist) == len(labellist))
    return senlist, labellist


sentences, labels = read_data("data_train.csv")
char_set = set(word for sen in sentences for word in sen)
char_dic = {j: i + 1 for i, j in enumerate(char_set)}
char_dic["unk"] = 0


def process_data(data, labels, dic, maxlen):
    sen2id = [[dic.get(char, 0) for char in sen] for sen in data]
    labels = np_utils.to_categorical(labels)
    return pad_sequences(sen2id, maxlen=maxlen), labels


train_data, train_labels = process_data(sentences, labels, char_dic, 100)


# squash压缩函数和原文不一样，可自己定义
def squash(x, axis=-1):
    # K.square:逐项平方; K.sqrt:逐项开方; K.epsilon:以数值形式返回一个（一般来说很小的）数，用以防止除0错误
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale  # S/||S||


class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        # 动态路由部分
        for i in range(self.routings):
            # K.premute_dimensions: 按照给定的模式重排一个张量的轴
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


def build_model(vocab, emb_dim, maxlen, n_cap, cap_dim, n_class):
    word_input = Input(shape=(None,), dtype="int32")
    embed = Embedding(input_dim=len(vocab),
                      output_dim=100,
                      input_length=maxlen
                      )(word_input)
    x = Bidirectional(LSTM(100, return_sequences=True))(embed)
    x = Capsule(num_capsule=n_cap, dim_capsule=cap_dim, routings=3, share_weights=True)(x)
    # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_class, activation='softmax')(x)
    model = Model(inputs=word_input, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.summary()
    return model


model = build_model(char_dic, 100, 200, 100, 100, 3)

model.fit(train_data, train_labels, batch_size=16, epochs=3, validation_split=0.2)
