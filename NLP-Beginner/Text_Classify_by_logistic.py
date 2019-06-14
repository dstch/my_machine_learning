import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def bag_of_word(words):
'''
@description: 将文本数据转换为词袋模型向量
@param {type} 
@return: 
'''
    word_list = [x.split() for x in words]
    word_list=set(word_list)
    for line in words:
        

def vocabulary_of_word(words):
'''
@description: 将文本数据转换为词汇表模型向量
@param {type} 
@return: 
'''    


data = pd.read_csv('data/train.tsv',usecols=[3])
data_x_list=np.array(data).tolist()