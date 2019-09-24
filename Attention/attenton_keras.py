#!/usr/bin/env python
# encoding: utf-8
"""
@author: dstch
@license: (C) Copyright 2013-2019, Regulus Tech.
@contact: dstch@163.com
@file: attenton_keras.py
@time: 2019/5/16 19:43
@desc: attention by keras
"""

from keras.layers import Bidirectional, CuDNNLSTM, Embedding, Input, SpatialDropout1D, Dense, add, GlobalMaxPooling1D, \
    GlobalAveragePooling1D, concatenate, Layer
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import initializers, regularizers, constraints
from keras import backend as K

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
    '''
    credits go to: https://www.kaggle.com/thousandvoices/simple-lstm/
    '''
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)

    words = Input(shape=(MAX_LEN,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    # ---------------------------------- attention -----------------------------
    att = Attention(MAX_LEN)(x)

    hidden = concatenate([att, GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x), ])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)

    model = Model(inputs=words, outputs=result)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

    # loss如果是列表，则模型的output需要是对应的列表
    # model.compile(loss=[custom_loss, 'binary_crossentropy'], optimizer='adam', metrics=["accuracy"])

    # clr
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


def my_load_model(model_path, test_data):
    print('[INFO] loading network...')

    # 在载入模型时，显示声明自定义层
    model = load_model(model_path, custom_objects={'Attention': Attention})

    prediction = model.predict(test_data, batch_size=BATCH_SIZE)

    # 此处是模型直接输出，后续可以采用LabelBinarizer将数值转换为实际标签
    return prediction
