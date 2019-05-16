#!/usr/bin/env python
# encoding: utf-8
"""
@author: dstch
@license: (C) Copyright 2013-2019, Regulus Tech.
@contact: dstch@163.com
@file: clr_keras.py
@time: 2019/5/15 11:49
@desc: Cyclical Learning Rate in Keras
https://blog.csdn.net/zzc15806/article/details/79711114
"""

from keras.layers import Bidirectional, CuDNNLSTM, Embedding, Input, SpatialDropout1D, Dense, add, GlobalMaxPooling1D, \
    GlobalAveragePooling1D, concatenate, Layer
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import initializers, regularizers, constraints
from keras import backend as K
import keras

EMB_SIZE = 300
MAX_LEN = 220
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 768
FOLD_NUM = 3
OOF_NAME = 'predicted_target'
BATCH_SIZE = 1024
PATIENCE = 3


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


def build_model(embedding_matrix, X_train, y_train, X_valid, y_valid):
    """
    clr demo by customized function
    :param embedding_matrix:
    :param X_train:
    :param y_train:
    :param X_valid:
    :param y_valid:
    :return:
    """

    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)

    words = Input(shape=(MAX_LEN,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    att = Attention(MAX_LEN)(x)
    hidden = concatenate([att, GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x), ])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)

    model = Model(inputs=words, outputs=result)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

    # loss如果是列表，则模型的output需要是对应的列表
    # model.compile(loss=[custom_loss, 'binary_crossentropy'], optimizer='adam', metrics=["accuracy"])

    # ------------------------ clr ----------------------------------------
    def scheduler(epoch):
        # 每隔100个epoch，学习率减小为原来的1/10
        if epoch % 100 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        return K.get_value(model.optimizer.lr)

    reduce_lr = LearningRateScheduler(scheduler)

    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=2, validation_data=(X_valid, y_valid),
              verbose=1, callbacks=[early_stop, reduce_lr])  # use clr
    # aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
    #
    # model = Model(inputs=words, outputs=[result, aux_result])
    # model.compile(loss=[custom_loss, 'binary_crossentropy'], loss_weights=[loss_weight, 1.0], optimizer='adam')

    return model


def keras_clr_by_native_method():
    """
    以下方法都用于model.fit(callbacks=[...])，控制训练过程
    :return:
    """
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
                                      epsilon=0.0001, cooldown=0, min_lr=0)
    """
    当评价指标不在提升时，减少学习率
    当学习停滞时，减少2倍或10倍的学习率常常能获得较好的效果。该回调函数检测指标的情况，如果在patience个epoch中看不到模型性能
    提升，则减少学习率参数

    参数
    monitor：被监测的量
    factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
    patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
    mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。
    epsilon：阈值，用来确定是否进入检测值的“平原区”
    cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
    min_lr：学习率的下限
    """

    schedule = None
    keras.callbacks.LearningRateScheduler(schedule)
    """
    该回调函数是学习率调度器

    参数
    schedule：函数，该函数以epoch号为参数（从0算起的整数），返回一个新学习率（浮点数）
    """

    keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
    """
    当监测值不再改善时，该回调函数将中止训练

    参数
    monitor：需要监视的量
    patience：当early stop被激活（如发现loss相比上一个epoch训练没有下降），则经过patience个epoch后停止训练。
    verbose：信息展示模式
    mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值停止下降则中止训练。在max模式下，当检测值不再上升则停止训练。
    """