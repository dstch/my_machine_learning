#!/usr/bin/env python
# encoding: utf-8
"""
@author: dstch
@license: (C) Copyright 2013-2019, Regulus Tech.
@contact: dstch@163.com
@file: multi-class.py
@time: 2019/8/22 15:04
@desc:
"""
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

y_true = []
y_pred = []

# 计算混淆矩阵并可视化
matrix = confusion_matrix(y_true, y_pred)
df_matrix = pd.DataFrame(matrix)
plt.figure(figsize=(5.5, 4))
sns.heatmap(df_matrix, annot=True)
plt.title('Multi-Class Confusion Matrix \nAccuracy：{0:.3f}'.format(accuracy_score(y_true, y_pred)))
plt.ylabel('True label')
plt.xlabel('Prediction label')

# 多分类评估报告
print(classification_report(y_true, y_pred))

# 宏平均和微平均
# 宏平均和微平均的对比
#
#     如果每个class的样本数量差不多,那么宏平均和微平均没有太大差异
#     如果每个class的样本数量差异很大,而且你想:
#         更注重样本量多的class:使用微平均
#         更注重样本量少的class:使用宏平均
#     如果微平均大大低于宏平均,检查样本量多的class
#     如果宏平均大大低于微平均,检查样本量少的class
precision_score(y_true, y_pred, average='micro')  # 微平均
precision_score(y_true, y_pred, average='macro')  # 宏平均
