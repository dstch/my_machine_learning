#!/usr/bin/env python
# encoding: utf-8
"""
@author: dstch
@license: (C) Copyright 2013-2019, Regulus Tech.
@contact: dstch@163.com
@file: jieba_log.py
@time: 2019/6/27 20:05
@desc: https://zhuanlan.zhihu.com/p/48382440
    让jieba闭嘴
"""

# 第一种
import jieba
import logging

logger = logging.getLogger()
# 配置 logger
jieba.default_logger = logger
tokenizer = jieba.Tokenizer()
tokenizer.initialize()

# 第二种
import jieba

for handler in jieba.default_logger.handlers:
    jieba.default_logger.removeHandler(handler)
tokenizer = jieba.Tokenizer()
tokenizer.initialize()
