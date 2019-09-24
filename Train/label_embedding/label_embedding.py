#!/usr/bin/env python
# encoding: utf-8
"""
@author: dstch
@license: (C) Copyright 2013-2019, Regulus Tech.
@contact: dstch@163.com
@file: label_embedding.py
@time: 2019/9/24 17:03
@desc: 在训练中，对分类文本标签进行处理的相关方法
"""

from numpy import np
import pickle
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer

# MultiLabelBinarizer是针对多个标签，比如标签为 ['蓝色','上衣']
# LabelBinarizer是针对常见的单标签

lb_path = ''  # 标签转换器保存位置

labels = []  # 从数据集中得到的标签列表

lb = LabelBinarizer()  # 初始化标签embedding

labels = np.array(labels)
oh_labels = lb.fit_transform(labels)  # 对标签列表进行embedding，之后可以直接放入模型进行训练

prediction = []  # 假设prediction是从model.predict得到的预测结果

str_labels = lb.inverse_transform(prediction)  # 将模型输出转换为标签

# 保存标签转换器，以备部署使用
f = open(lb_path, 'wb')
f.write(pickle.dumps(lb))
f.close()

# 读取
lb = pickle.load(open(lb_path, 'rb').read())

dic = lb.class_  # 可以得到标签列表
