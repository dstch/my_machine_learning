#!/usr/bin/env python
# encoding: utf-8
"""
@author: dstch
@license: (C) Copyright 2013-2019, Regulus Tech.
@contact: dstch@163.com
@file: clr_keras.py
@time: 2019/5/15 11:49
@desc: Cyclical Learning Rate in Keras
"""

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
              verbose=1, callbacks=[early_stop, reduce_lr])
    # aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
    #
    # model = Model(inputs=words, outputs=[result, aux_result])
    # model.compile(loss=[custom_loss, 'binary_crossentropy'], loss_weights=[loss_weight, 1.0], optimizer='adam')

    return model