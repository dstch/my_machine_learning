import numpy as np
import pandas as pd


def bag_of_word(words):
    # 将文本数据转换为词袋模型向量
    word_list=[]
        

data=pd.read_csv('data/train.tsv')
