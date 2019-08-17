#!/usr/bin/env python
# encoding: utf-8
"""
@author: dstch
@license: (C) Copyright 2013-2019, Regulus Tech.
@contact: dstch@163.com
@file: Text_Classify_by_Keras.py
@time: 2019/8/16 16:48
@desc: 对数据进行分词、去停用词，对中文和英文分别进行embedding进行比较
"""

from keras.layers import Bidirectional, LSTM, CuDNNLSTM, Embedding, Input, Dense, add, GlobalAveragePooling1D, \
    GlobalMaxPool1D, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras import initializers, regularizers
from keras import backend as K
from gensim.models.word2vec import Word2Vec, LineSentence
from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing import text, sequence
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from keras.models import load_model
import pandas as pd
import sklearn.metrics

MAX_LEN = 80
PATIENCE = 3
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 512
BATCH_SIZE = 1024
EMBEDDING_DIM = 100


def read_data(file):
    """
    从csv文件中读取数据
    :param file:
    :return:
    """
    df = pd.read_csv(file, encoding='utf-8')
    c_texts = []
    e_texts = []
    labels = []
    for index, row in df.iterrows():
        c_texts.append(row['caseName'])
        e_texts.append(row['message'])
        labels.append(row['result'])
    return c_texts, e_texts, labels


def create_embedding(word_index, word2vec_model):
    # https: // github.com / SophonPlus / ChineseWordVectors
    embedding_matrix = np.zeros(word2vec_model.wv.vectors.shape)
    for word, i in word_index.items():
        try:
            embedding_vector = word2vec_model[word]
            embedding_matrix[i] = embedding_vector
        except:
            continue
    return embedding_matrix


def create_glove_embedding(glove_path, word_index):
    # 读取glove文件
    embeddings_index = {}
    with open(glove_path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    # 构建embedding矩阵
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def build_model(embedding_matrix, X_train, y_train, X_val, y_val):
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)

    words = Input(shape=(MAX_LEN,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)
    hidden = concatenate([GlobalMaxPool1D()(x), GlobalAveragePooling1D()(x)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(y_train.shape[-1], activation='sigmoid')(hidden)
    model = Model(input_shape=words, outputs=result)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

    def scheduler(epoch):
        if epoch % 100 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        return K.get_value(model.optimizer.lr)

    reduce_lr = LearningRateScheduler(scheduler)
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=4, validation_data=(X_val, y_val), verbose=1,
              callbacks=[early_stop, reduce_lr])
    return model


def train():
    file = ''
    w2v_model = ''
    split_data_file = ''
    model = Word2Vec.load(w2v_model)
    _, _, labels = read_data(file)
    mlb = LabelBinarizer()
    labels = np.array(labels)
    labels = mlb.fit_transform(labels)
    texts = read_split_data(split_data_file)
    X_trian, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, stratify=labels)

    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(list(X_trian) + list(X_test))
    x_trian = tokenizer.texts_to_sequences(X_trian)
    x_test = tokenizer.texts_to_sequences(X_test)
    x_trian = sequence.pad_sequences(x_trian, maxlen=MAX_LEN)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
    embedding_matrix = create_embedding(tokenizer.word_index, model)

    train_model = build_model(embedding_matrix, x_trian, y_train, x_test, y_test)
    train_model.save('')


def predict():
    file = ''
    w2v_model = ''
    split_data_file = ''
    _, _, labels = read_data(file)
    mlb = LabelBinarizer()
    labels = np.array(labels)
    labels = mlb.fit_transform(labels)
    texts = read_split_data(split_data_file)
    X_trian, X_test, _, y_test = train_test_split(texts, labels, test_size=0.2, stratify=labels)

    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(list(X_trian) + list(X_test))
    x_test = tokenizer.texts_to_sequences(X_test)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

    model = load_model('')
    prediction = model.predict(x_test)
    pre_labels = mlb.inverse_transform(prediction)
    str_labels = mlb.inverse_transform(y_test)
