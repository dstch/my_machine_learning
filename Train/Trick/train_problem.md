# keras训练性能调优方法

## 1、val_loss曲线或val_acc曲线震荡不平滑

原因可能如下：

        学习率可能太大
        batch size太小
        样本分布不均匀
        缺少加入正则化
        数据规模较小

## 2、val_acc几乎为0

一种很重要的原因是数据split的时候没有shuffle

        import numpy as np
        index = np.arange(data.shape[0])
        np.random.seed(1024)
        np.random.shuffle(index)
        data=data[index]
        labels=labels[index]

## 3、训练过程中loss数值为负数？

原因：输入的训练数据没有归一化造成
解决方法：把输入数值通过下面的函数过滤一遍，进行归一化

        #数据归一化  
        def data_in_one(inputdata):  
                inputdata = (inputdata-inputdata.min())/(inputdata.max()-inputdata.min())
                return inputdata 

## 4、怎么看loss和acc的变化（loss几回合就不变了怎么办？）

        train loss 不断下降，test loss不断下降，说明网络仍在学习;
        train loss 不断下降，test loss趋于不变，说明网络过拟合;
        train loss 趋于不变，test loss不断下降，说明数据集100%有问题;
        train loss 趋于不变，test loss趋于不变，说明学习遇到瓶颈，需要减小学习率或批量数目;
        train loss 不断上升，test loss不断上升，说明网络结构设计不当，训练超参数设置不当，数据集经过清洗等问题。

## 5、训练中发现loss的值为NAN，这时可能的原因如下：

    1）学习率太高
    2）如果是自己定义的损失函数，这时候可能是你设计的损失函数有问题

    一般来说，较高的acc对应的loss较低，但这不是绝对，毕竟他们是两个不同的东西，所以在实际实现中，我们可以对两者进行一个微调。

## 6、epoch轮数/BN/dropout/

    关于epoch设置问题，我们可以设置回调函数，选择验证集最高的acc作为最优模型。

    关于BN和dropout，其实这两个是两个完全不同的东西，BN针对数据分布，dropout是从模型结构方面优化，所以他们两个可以一起使用，对于BN来说其不但可以防止过拟合，还可以防止梯度消失等问题，并且可以加快模型的收敛速度，但是加了BN，模型训练往往会变得慢些。

## 7、深度网络的过拟合问题讨论

### 7.1、加入dropout层

        ……
        from keras.layers import Concatenate,Dropout
        ……
        concatenate = Concatenate(axis=2)([blstm,embedding_layer])

        concatenate=Dropout(rate=0.1)(concatenate)

### 7.2、检查数据集是否过小（Data Augmentation）

        一个比较合理的扩增数据集的方法就是将每一个文本的句子循环移位，这样可以最大限度地保证文本整体的稳定

### 7.3、用迁移学习的思想

        具体来讲就是model.load人家训练好的weight.hdf5，然后在这个基础上继续训练。具体可以见之后的博文中的断点训练。

### 7.4、调参小tricks

        调小学习速率（Learning Rate）
        适当增大batch_size。
        试一试别的优化器（optimizer）
        Keras的回调函数EarlyStopping() 

### 7.5、正则化方法

        正则化方法是指在进行目标函数或代价函数优化时，在目标函数或代价函数后面加上一个正则项，一般有L1正则与L2正则等。

代码片段示意：

        from keras import regularizers
        ……
        out = TimeDistributed(Dense(hidden_dim_2,
                            activation="relu",
                            kernel_regularizer=regularizers.l1_l2(0.01,0.01),
                            activity_regularizer=regularizers.l1_l2(0.01,0.01)
                            )
                      )(concatenate)

        ……

        dense=Dense(200,
            activation="relu",
            kernel_regularizer=regularizers.l1_l2(0.01,0.01),
            activity_regularizer=regularizers.l1_l2(0.01,0.01)
            )(dense)
