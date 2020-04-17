# TensorFlow v2.0使用案例

>***TensorFlow主要用于从K1~Y1阶段的还款预测，下文会介绍模型搭建的详细过程。***

***

## 1. 神经网络基本原理介绍

>如图是一个单隐层的神经网络。
>![单隐层神经网络](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1586929438960&di=2a2fddf5f183796f0ee403fa58a99937&imgtype=0&src=http%3A%2F%2Fstatic.leiphone.com%2Fuploads%2Fnew%2Farticle%2F740_740%2F201705%2F59267d72460ba.png)
>
>---
>
>### 1.1 基本结构介绍
>
>>***神经网络的结构千变万化，这里只介绍全连接的顺序神经网络。神经网络的层结构主要可以归为３类：输入层、输出层以及中间层（包含各类网络结构），各个神经层由一个个神经元组成。***
>>
>>---
>>
>>#### <1>　输入层
>>>输入层是我们的数据入口，每个神经元都代表一个输入特征（通常来说，进入输入层的特征都需要预先进行一些特征处理：标准化、onehot处理等）
>>
>>#### <2>　中间层
>>>中间层是数据处理的主要过程，例如全连接层网络的中间层可以提高模型对非线性数据的拟合能力；卷积层可以提取结构化数据的局部结构信息等等。
>>
>>#### <3>　输出层
>>>输出层是将模型的拟合结果以我们需要的方式输出，例如回归时输出回归值；分类时输入各类概率等。
>
>---
>
>### 1.2 前向传播算法
>>前向传播算法是神经网络集基本计算方法，是由输入到输出过程中的具体计算方法。
>>
>>---
>>
>>#### <1>　层间权值
>>
>>>相邻两个层之间，任意两个不同层的神经元之间可以存在连线，每条连线都对应着一个权值$w_{ij}$。
>>#### <2>　偏置项
>>>除了输入层以外，通常每个神经元都会存在一个偏置项$b_i$。
>>#### <3>　激活函数
>>>除了输入以及输出层，通常每个神经元都会有一个激活函数（通常同一层的神经元激活函数相同），激活函数的主要目的为加强模型对非线性数据的拟合能力，设激活函数为sigmoid函数，其具体公式如下：
>>>$$
>>>\sigma(x) = \frac {1}{1+e^{-x}}
>>>$$
>>
>>---
>>
>>#### 前向算法计算方式
>>这里以图中的单隐层神经网络为例，展示前向算法的计算方式。
>>##### 输入层：
>>>每个神经元$i$对应着一个特征$X_i$。
>>>神经元$i$的输出为：
>>>$$
>>>x_i=特征X_i的输入值
>>>$$
>>##### 隐层：
>>>每个神经元$j$与输入层神经元$i$对应的权值为$w_{ij}^{(2)}$，对应的偏置项为$b_j^{(2)}$。
>>>神经元$j$的输入为：
>>>$$
>>>z_j^{(2)} = \sum_{i=1}^{n}w_{ij}^{(2)}x_i+b_j^{(2)}
>>>$$
>>>神经元$j$的输出为:
>>>$$
>>>a_j^{(2)} = \sigma(z_j^{(2)})
>>>$$
>>##### 输出层：
>>
>>>每个神经元$k$与隐层神经元$j$对应的权值为$w_{jk}^{(3)}$，对应的偏置项为$b_k^{(3)}$。
>>>神经元$k$的输入为：
>>>$$
>>>z_k^{(3)} = \sum_{j=1}^{n}w_{jk}^{(3)}a_j^{(2)}+b_k^{(3)}
>>>$$
>>>神经元$k$的输出为：
>>>$$
>>>\hat y_k = a_k^{(3)} = \sigma(z_k^{(3)})
>>>$$
>>
>>---
>>
>
>### 1.3 损失逆传播算法（BP算法）
>>BP算法是参数更新算法，对于给定的损失函数，沿着损失函数的负梯度方向更新参数，以达到训练模型的效果。
>>#### 损失函数
>>常用的损失函数有均方误差、交叉熵损失函数、绝对值损失函数等，具体公式如下：
>>>均方误差(MSE)：
>>>$$
>>>MSE = \frac {1}{N}\sum_{i=1}^N(\hat y_i-y_i)^2
>>>$$
>>>绝对值损失函数(MAE)：
>>>$$
>>>MAE = \frac {1}{N}\sum_{i=1}^N|\hat y_i-y_i|
>>>$$
>>>交叉熵损失函数(Cross Entropy Error)：
>>>$$
>>>CrossEntropy = -\frac {1}{N}\sum_{i=1}^N\sum_{k=1}^Ty_{ik}log\hat y_{ik}
>>>$$
>>>其中$y_i$是第$i$条数据的真实值，$\hat y_i$是对第$i$条数据的预测值，$y_{ik}$是真实标签的转为onehot格式后第$i$条数据的第k类标签的值，$\hat y_{ik}$是对第$i$条数据的第$k$类标签的预测概率，通常是$softmax$函数的输出。
>>>
>>>通常做回归时，用的损失函数是$MSE/MAE$，分类时常用的损失函数是$CrossEntropy$。
>>
>>---
>>
>>#### 参数更新算法
>>
>>参数更新是基于损失函数的，目的是参数更新后可以使得损失函数下降。为了方便计算与阅读，下面举例说明，损失函数选用$MSE$，更新的参数选取$w_{jk}^{(3)}$。
>>
>>> 基于前向算法，可以得到以下公式：
>>> $$
>>> z_k^{(3)} = \sum_{j=1}^{n}w_{jk}^{(3)}a_j^{(2)}+b_k^{(3)}\tag{1}
>>> $$
>>> $$
>>> \hat y_k = \sigma(z_k^{(3)}) = \frac {1}{1+e^{-z_k^{(3)}}}\tag{2}
>>> $$
>>>
>>> 损失函数：
>>> $$
>>> L = \frac {1}{2}(y_k-\hat y_k)^2\tag{3}
>>> $$
>>> 参数$w_{jk}^{(3)}$关于$L$的梯度为：
>>> $$
>>> \frac {\partial L}{\partial w_{jk}^{(3)}}=\frac {\partial L}{\partial \hat y_k}*\frac {\partial \hat y_k}{\partial z_k^{(3)}}*\frac {\partial z_k^{(3)}}{\partial w_{jk}^{(3)}}\tag{4}
>>> $$
>>> 所以参数的更新公式是：
>>> $$
>>> \Delta w_{jk}^{(3)}=-\eta*\frac {\partial L}{\partial \hat y_k}\tag{5}
>>> $$
>>> 其中$\eta$是学习率。
>>>
>>> 由于$sigmoid$函数有如下特性：
>>> $$
>>> \sigma'(x)=(\frac {1}{1+e^{-x}})'=\sigma(x)*(1-\sigma(x))
>>> $$
>>> 所以根据式$(1)、(2)、(3)$将式$(4)$改写为：
>>> $$
>>> \frac {\partial L}{\partial w_{jk}^{(3)}}=(\hat y_k-y_k)*\hat y_k*(1-\hat y_k)*a_j^{(2)}
>>> $$
>>> 参数$w_{jk}^{(3)}$的更新公式为：
>>> $$
>>> \Delta w_{jk}^{(3)}=-\eta*\hat y_k*(\hat y_k-y_k)*(1-\hat y_k)*a_j^{(2)}
>>> $$
>>> 其他参数更新方法类似，就不再重复书写。、
>
>神经网络的基本原理大致就是上文所描述样子，如果感兴趣可以上网寻找一些特殊的网络结构自行学习。

---

## 2. TensorFlow v2.0搭建神经网络模型
>TensorFlow v2.0是一个全新的TensorFlow版本，语句与v1.x版本有较多区别，下面就给予现有数据进行建模，代码展示如下：
><1>　载入需要的包以及数据（数据为M2+的中原客户及其还款标识）：
>>```python
>>##载入所需包　
>>import pandas as pd
>>import numpy as np
>>import tensorflow as tf
>>
>>##载入数据
>>data = pd.read_csv('./PycharmProjects/M1_pay/data/train.csv').fillna(0)
>>data = data.iloc[:, 2:]
>>```
>>data的数据格式如下：
>>| pay_status | install_cnt | install_total_cnt | install_rate | ...  | login_status |
>>| :--------: | :---------: | :---------------: | :----------: | :--: | :----------: |
>>|     0      |      9      |        12         |   0.013173   | ...  |      0       |
>>|     0      |      6      |         6         |   0.012865   | ...  |      0       |
>>|     0      |     12      |        12         |   0.014154   | ...  |      0       |
>>|     0      |     12      |        12         |   0.011944   | ...  |      1       |
>>|    ...     |     ...     |        ...        |     ...      | ...  |     ...      |
>>|     0      |      3      |        12         |   0.017128   | ...  |      14      |
>
><2>　提取标签及特征，并对特征进行标准化（Z-score标准化）：
>>```python
>>##标签提取、特征提取及其标准化
>>label = data['pay_status']
>>feature = data.iloc[:,1:]
>>feature = (feature - feature.mean())/feature.std()
>>```
>>处理后的feature如下所示：
>>| install_cnt | install_total_cnt | install_rate | trans_amt | ...  | login_status |
>>| :---------: | :---------------: | :----------: | :-------: | :--: | :----------: |
>>|  0.412245   |     0.358537      |  -0.158954   | 0.572258  | ...  |  -0.152884   |
>>|  -0.419700  |     -2.183270     |  -0.243721   | -0.344891 | ...  |  -0.152884   |
>>|  1.244190   |     0.358537      |   0.111036   | 2.008793  | ...  |  -0.152884   |
>>|  1.244190   |     0.358537      |  -0.497197   | -0.230755 | ...  |  -0.059252   |
>>|     ...     |        ...        |     ...      |    ...    | ...  |     ...      |
>>|  -1.251645  |     0.358537      |   0.929537   | -0.367002 |      |   1.157963   |
>
><3>　划分测试集与训练集：
>>```python
>>##确定训练集、测试集索引
>>test_num = int(data.shape[0] * 0.2)
>>random_index = np.random.permutation(data.index)
>>test_index, train_index = random_index[:test_num], random_index[test_num:]
>>
>>##确定测试集、训练集
>>test_label, train_label = np.array(label[test_index]).astype('int32'),\
>>                                                  np.array(label[train_index]).astype('int32')
>>test_feature, train_feature = np.array(feature.iloc[test_index,:]).astype('float32'),\
>>                                                           np.array(feature.iloc[train_index,:]).astype('float32')
>>
>>##一个训练批次的数据量为32条(数量越大内存占用越高)
>>dtrain = tf.data.Dataset.from_tensor_slices((train_feature, train_label)).batch(32)
>>dtest = tf.data.Dataset.from_tensor_slices((test_feature, test_label)).batch(32)
>>```
>
><4>　搭建网络结构：
>>```python
>>##设计模型结构
>>class Model(tf.keras.Model):
>>        
>>        def __init__(self):
>>             super().__init__()
>>             self.d1 = tf.keras.layers.Dense(128, activation = 'relu')
>>             self.d2 = tf.keras.layers.Dense(256, activation = 'relu')
>>             self.d3 = tf.keras.layers.Dense(64, activation = 'relu')
>>             self.out = tf.keras.layers.Dense(2, activation = 'softmax')
>>        
>>        def call(self, feature):
>>             d1 = self.d1(feature)
>>             d2 = self.d2(d1)
>>             d3 = self.d3(d2)
>>             return self.out(d3)
>>
>>model = Model() 
>>```
>>模型结构如下图所示：
>>![网络结构](https://wx3.sinaimg.cn/mw690/00872OYVgy1gdun0gtx0nj32801o04fx.jpg)
>
><5>　设置损失函数、梯度下降优化器以及评估函数
>>```python
>>##设置损失函数以及梯度下降优化器
>>loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
>>optimizer = tf.keras.optimizers.Adam(0.00005)
>>
>>##设置模型评估函数
>>train_loss = tf.keras.metrics.Mean(name='train_loss') 
>>train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
>>test_loss = tf.keras.metrics.Mean(name='test_loss') 
>>test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
>>```
>
><6>　设置训练步骤以及测试步骤
>>```python
>>##设置模型训练步骤以及测试步骤
>>@tf.function
>>def train_step(feature, label):
>>        with tf.GradientTape(persistent=True) as tape:
>>             pred = model(feature)
>>             loss = loss_object(label, pred)
>>        gradients = tape.gradient(loss, model.trainable_variables)
>>        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
>>        del tape
>>    
>>        train_loss(loss)
>>        train_accuracy(label, pred)
>>    
>>@tf.function
>>def test_step(feature, label):
>>        pred = model(feature)
>>        loss = loss_object(label, pred)
>>    
>>        test_loss(loss)
>>        test_accuracy(label, pred)   
>>```
>
><7>　训练模型
>>```python
>>##模型训练
>>epoch = 50
>>for step in range(epoch):
>>   train_loss.reset_states()
>>   train_accuracy.reset_states()
>>   test_loss.reset_states()
>>   test_accuracy.reset_states() 
>>   
>>   for feature, label in dtrain:
>>        train_step(feature, label)
>>   
>>   for feature, label in dtest:
>>        test_step(feature, label)
>>        
>>   out = '步数:{}, 训练误差:{:.6f}, 训练准确率:{:.2f}%, 测试误差:{:.6f}, 测试准确率:{:.2f}%'
>>   print(out.format(step+1,
>>                     train_loss.result(),
>>                     train_accuracy.result()*100,
>>                     test_loss.result(),
>>                     test_accuracy.result()*100))
>>```
>>结果展示如下：
>>```python
>>步数:1, 训练误差:0.449160, 训练准确率:80.35%, 测试误差:0.425496, 测试准确率:81.96%
>>步数:2, 训练误差:0.401750, 训练准确率:82.52%, 测试误差:0.417339, 测试准确率:82.35%
>>步数:3, 训练误差:0.393825, 训练准确率:82.90%, 测试误差:0.412313, 测试准确率:82.46%
>>步数:4, 训练误差:0.388578, 训练准确率:83.13%, 测试误差:0.408423, 测试准确率:82.67%
>>步数:5, 训练误差:0.384715, 训练准确率:83.25%, 测试误差:0.405678, 测试准确率:82.86%
>>步数:6, 训练误差:0.381669, 训练准确率:83.34%, 测试误差:0.403337, 测试准确率:82.83%
>>步数:7, 训练误差:0.379078, 训练准确率:83.49%, 测试误差:0.401513, 测试准确率:82.90%
>>步数:8, 训练误差:0.376827, 训练准确率:83.62%, 测试误差:0.399958, 测试准确率:82.98%
>>步数:9, 训练误差:0.374745, 训练准确率:83.72%, 测试误差:0.398721, 测试准确率:83.06%
>>...
>>步数:49, 训练误差:0.317561, 训练准确率:86.64%, 测试误差:0.396585, 测试准确率:83.68%
>>步数:50, 训练误差:0.316257, 训练准确率:86.73%, 测试误差:0.396989, 测试准确率:83.68%
>>```
>
><8>　验证模型效果
>
>>```python
>>##计算KS值
>>import sys 
>>sys.path.append('./Metric')
>>import KS_value as ks
>>test_pred = np.array(model(test_feature))[:,1]
>>ks_value = ks.KS_value(test_label, test_pred, 1)
>>```
>>输出结果如下：
>>```python
>>0.501792252983812
>>```
>
><9>　模型保存
>
>>```python
>>model.save_weights('./test_weight')
>>```
>
><10>　模型调用
>>```python
>>model1 = Model()
>>model1.load_weights('./test_weight')
>>model1(test_feature)
>>```
>>输出结果为：
>>```
>><tf.Tensor: id=147894, shape=(9263, 2), dtype=float32, numpy=
>>array([[0.90980166, 0.09019838],
>>       [0.84081537, 0.1591847 ],
>>       [0.8158288 , 0.18417121],
>>       ...,
>>       [0.985545  , 0.01445498],
>>       [0.99703753, 0.00296255],
>>       [0.8493371 , 0.15066287]], dtype=float32)>
>>```
>
>以上就是TensorFlow v2.0建立模型的全部过程，当然还有更简易的建模过程，集成度高，但是可配置化偏低，可以按照自己的需求选择建模方式。

***
<p align='right'>Author : Junyu
<p align='right'>Date : 2020-04-14