#!/usr/bin/env python
# encoding: utf-8
"""
@author: dstch
@license: (C) Copyright 2013-2019, Regulus Tech.
@contact: dstch@163.com
@file: model_save.py
@time: 2019/8/16 10:02
@desc: https://www.jianshu.com/p/0711f9e54dd2
"""

from keras.callbacks import Tensorboard

# keras.callbacks在model.fit中发挥作用,写法是:
......
tensorboard = Tensorboard(log_dir='log(就是你想存事件的文件夹)')
callback_lists = [tensorboard]  # 因为callback是list型,必须转化为list
model.fit(x_train, y_train, bach_size=batch_size, epochs=epoch, shuffle='True', verbose='True',
          callbacks=callback_lists)

# 保存结果最好的模型

from keras.callbacks import ModelCheckpoint

# checkpoint添加到上文的callback_lists列表中使用
checkpoint = ModelCheckpoint(filepath=file_name,  # (就是你准备存放最好模型的地方),
                             monitor='val_acc',  # (或者换成你想监视的值,比如acc,loss,
                             val_loss,  # 其他值应该也可以,还没有试),
                             verbose=1,  # (如果你喜欢进度条,那就选1,如果喜欢清爽的就选0,verbose=冗余的),
                             save_best_only='True',  # (只保存最好的模型,也可以都保存),
                             mode='auto',
                             # (如果监视器monitor选val_acc, mode就选'max',如果monitor选acc,mode也可以选'max',如果monitor选loss,mode就选'min'),一般情况下选'auto',
                             period=1)  # (checkpoints之间间隔的epoch数)
"""
filename：字符串，保存模型的路径
monitor：需要监视的值
verbose：信息展示模式，0或1
save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，
例如，当监测值为val_acc时，模式应为max，
当检测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
period：CheckPoint之间的间隔的epoch数
"""

# 再次使用模型的时候,需要调用

from keras.models import save_model  # (load_mode)

......
model = load_model('best_weights.h5')
loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)

print('test_loss', loss, 'test_accuracy', acc)

# 周期性学习率衰减
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001,
                                  cooldown=0, min_lr=0)
"""
参数
monitor：被监测的量
factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。
epsilon：阈值，用来确定是否进入检测值的“平原区”
cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
min_lr：学习率的下限
"""
