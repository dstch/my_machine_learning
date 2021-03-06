#!/usr/bin/env python
# encoding: utf-8
"""
@author: dstch
@license: (C) Copyright 2013-2019, Regulus Tech.
@contact: dstch@163.com
@file: train_metric.py
@time: 2019/8/21 11:27
@desc: 在keras训练（fit）过程中，加入其它评估参数
"""

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from Attention.attenton_keras import Attention
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.layers import Bidirectional, CuDNNLSTM, Embedding, Input, SpatialDropout1D, Dense, add, GlobalMaxPooling1D, \
    GlobalAveragePooling1D, concatenate, Layer
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import initializers, regularizers, constraints
from keras import backend as K
from keras.preprocessing import text
from gensim.models.word2vec import Word2Vec
import matplotlib.pyplot as plt

EMB_SIZE = 300
MAX_LEN = 220
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 768
BATCH_SIZE = 1024
PATIENCE = 3
EPOCHS = 10


class Metrics(Callback):
    def __init__(self, x_val, y_val):
        self.validation_data = [x_val, y_val]

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        # 多标签分类需要增加 average='micro'
        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average='micro')
        _val_precision = precision_score(val_targ, val_predict, average='micro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print('-val_f1: %.4f --val_precision: %.4f --val_recall: %.4f' % (_val_f1, _val_precision, _val_recall))
        return


def create_embedding(word_index, word2vec_model):
    """
    通过gensim训练的word2vec模型，构建词嵌入矩阵
    :param word_index:
    :param word2vec_model:
    :return:
    """
    embedding_matrix = np.zeros(word2vec_model.wv.vectors.shape)
    for word, i in word_index.items():
        try:
            embedding_vector = word2vec_model[word]
            embedding_matrix[i] = embedding_vector
        except:
            continue
    return embedding_matrix


def build_model(embedding_matrix, X_train, y_train, X_valid, y_valid):
    '''
    credits go to: https://www.kaggle.com/thousandvoices/simple-lstm/
    '''

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

    # Dense第一个参数是分类标签数
    result = Dense(y_train.shape[-1], activation='sigmoid')(hidden)

    model = Model(inputs=words, outputs=result)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

    return model


def scheduler(epoch):
    """
    clr
    :param epoch:
    :return:
    """
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 100 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)


texts = []  # corpus
labels = []
w2v_model = ''  # word2vec model path

# 标签处理，one-hot编码
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
# 把标签转回文本用 lb.inverse_transform()


# 数据集划分
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=2019)

# 文本数据处理
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(texts)
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)

model = Word2Vec.load(w2v_model)
embedding_matrix = create_embedding(tokenizer.word_index, model)
model = build_model(embedding_matrix, X_train, y_train, X_val, y_val)

# ---------------------------------- train metrics -----------------------------
early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)
reduce_lr = LearningRateScheduler(scheduler)
# fit方法返回history对象，可以从对象中获取训练过程的数据
H = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val),
              verbose=1, callbacks=[early_stop, reduce_lr, Metrics(X_val, y_val)])  # use clr

# 绘制训练过程的数据
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig('plot.png')
