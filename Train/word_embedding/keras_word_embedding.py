#!/usr/bin/env python
# encoding: utf-8
"""
@author: dstch
@license: (C) Copyright 2013-2019, Regulus Tech.
@contact: dstch@163.com
@file: keras_word_embedding.py
@time: 2019/9/25 11:16
@desc: keras在进行文本相关的训练前的通用文本处理
"""
import numpy as np
from keras.preprocessing import text, sequence
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.layers import Embedding, Input, LSTM, Bidirectional, Dense
from keras.models import Model


def creat_embedding(word_index, word2vec_model):
    embedding_matrix = np.zeros(word2vec_model.wv.vectors.shape)
    for word, i in word_index.items():
        try:
            embedding_vector = word2vec_model[word]
            embedding_matrix[i] = embedding_vector
        except:
            continue
    return embedding_matrix


def build_model(embedding_matrix, X_train, y_train, X_val, y_val):
    # 定义模型输入
    words = Input(shape=(MAX_LEN,))
    # 对文本进行embedding
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    # 进入神经网络
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)
    hidden = Dense(DENSE_HIDDEN_UNITS, activation='relu')(x)
    result = Dense(y_train.shape[-1], activation='sigmoid')(hidden)

    model = Model(inputs=words, outputs=result)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    H = model.fit(X_train, y_train, batch_size=128, epochs=50, validation_data=(X_val, y_val), verbose=1)

    return model, H


LSTM_UNITS = 256
DENSE_HIDDEN_UNITS = 256
MAX_LEN = 100
w2v_model = ''  # 词向量模型路径

# 数据来源于数据文件，格式为列表
texts = []  # 文本数据需要先进行分词
labels = []

# 标签进行二值化处理，具体可以参照label_embedding
lb = LabelBinarizer()
labels = lb.fit_transform(np.array(labels))

# 对数据集进行划分,test_size：划分比例, stratify：按照labels的比例进行划分
X_train, X_text, y_train, y_test = train_test_split(texts, labels, test_size=0.2, stratify=labels)

# 载入由gensim训练好的词向量预训练模型
word2vec_model = Word2Vec.load(w2v_model)

# 对文本进行序列化处理
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(texts)
x_train = tokenizer.texts_to_sequences(X_train)
x_test = tokenizer.texts_to_sequences(X_text)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
embedding_matrix = creat_embedding(tokenizer.word_index, word2vec_model)

# 构建模型
build_model(embedding_matrix, X_train, y_train, x_test, y_test)
