#!/usr/bin/env python
# encoding: utf-8
"""
@author: dstch
@license: (C) Copyright 2013-2019, Regulus Tech.
@contact: dstch@163.com
@file: keras_tensorboard.py
@time: 2019/5/17 0:08
@desc: use tensorboard in keras
"""

import keras

keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0,
                            embeddings_layer_names=None, embeddings_metadata=None)
"""
    log_dir：保存日志文件的地址，该文件将被TensorBoard解析以用于可视化
    histogram_freq：计算各个层激活值直方图的频率（每多少个epoch计算一次），如果设置为0则不计算。
    write_graph: 是否在Tensorboard上可视化图，当设为True时，log文件可能会很大
    write_images: 是否将模型权重以图片的形式可视化
    embeddings_freq: 依据该频率(以epoch为单位)筛选保存的embedding层
    embeddings_layer_names:要观察的层名称的列表，若设置为None或空列表，则所有embedding层都将被观察。
    embeddings_metadata: 字典，将层名称映射为包含该embedding层元数据的文件名，参考这里获得元数据文件格式的细节。
    如果所有的embedding层都使用相同的元数据文件，则可传递字符串。
"""
