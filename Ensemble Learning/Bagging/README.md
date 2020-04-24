# Bagging介绍

>***Bagging是一个并行式的集成学习框架，将独立的低偏差高方差的强学习器通过投票或者平均等方法以降低模型的方差。由于是并行式的，所以所有子学习器可以同步进行，时间复杂度为很低，计算速度相较于Boosting有很大提升，RandomForest算法是Bagging的扩展变体，对此下文将着重介绍该算法。***

---

## 1. RandomForest介绍

> RandomForest是Bagging的扩展变体，因为Bagging需要每个子学习器有较大差异，因此传统的Bagging算法是进行行抽样（即对数据样本进行抽样），不同的样本进行训练可以使得训练出的子学习器有较大差异。RandomForest算法在行抽样的基础上，还进行了列抽样（即特征抽样），使得每次训练的子学习器都侧重于不同的特征。
>
> ### 1.1　行抽样与列抽样
>
> ---
> #### 行采样
> 在RandomForest中，行抽样一般采用自助采样法（bootstrap sampling），其具体方法如下：
> >１．包含$m$个样本的数据集$D$，从中随机抽取一个样本$a$。
> >２．将样本$a$复制进采样数据集$D'$，并将样本$a$放回数据集$D$。
> >３．重复进行$m$次上述步骤，可以得到$m$个样本的采样数据集$D'$。
> >对于每个样本，始终不被采样到的概率为$(1-\frac{1}{m})^m$，对其取极限得到：
> >$$
> >\underset{x\to \infty}{\lim}(1-\frac{1}{m})^m=\frac{1}{e}\approx0.368
> >$$
> >所以如果采用自助采样法，可以使得36.8%的样本不被采样到，未被采样到的数据作为测试集，采样得到的作为训练集。
>
> ---
>
> #### 列采样
> RandomForest的列采样是在每次分裂节点时进行的，具体方法如下：
>
> >１．从待分裂节点的特征集合（包含d个特征）中随机选取k个特征。
> >２．从该k个特征之间选取最优的分裂特征进行划分。
> >k为随机性的引入程度，如果k=d则随机森林退化为传统的Bagging算法。
>
> ---
>
> ### 1.2　RandomForest的结合方法。
>
> 通常来说，RandomForest对分类问题使用投票法；对回归问题使用平均法，具体方法如下：
>
> > #### 投票法
> > $$
> > H(x)=\underset{y \in Y}{\arg\max}\sum_{t=1}^TI(h_t(x)=y)
> > $$
> > 其中$T$代表子学习器的数量，$Y$代表分类的类数。
> >
> > #### 平均法
> > $$
> > H(x)=\frac{1}{T}\sum_{t=1}^Th_t(x)
> > $$
> 
>
>
> RandomForest的基本原理就讲到这里，想更深入了解可以查阅相关文件。

---

## 2. 示例代码
>安装sklearn包
>```python
>pip install sklearn
>```
>
><1>　载入所需包及数据
>
>```python
>##载入所需包
>from sklearn.ensemble import RandomForestClassifier
>import pandas as pd
>import numpy as np
>import joblib
>
>##载入训练数据
>data = pd.read_csv('./PycharmProjects/M1_pay/data/train.csv').fillna(0)
>```
>
><2>　数据集拆分
>```python
>label = data['pay_status']
>feature = data.iloc[:, 3:]
>train_rate = 0.7
>random_index = np.random.permutation(feature.shape[0])
>train_index, test_index = random_index[:int(train_rate * feature.shape[0])], \
>                                random_index[int(train_rate * feature.shape[0]):]
>train_label, test_label = label[train_index], label[test_index]
>train_feature, test_feature = feature.iloc[train_index, :], feature.iloc[test_index, :]
>```
>
><3>　设置模型参数
>```python
>model = RandomForestClassifier(n_estimators = 50, oob_score = True, verbose = 0)
>```
>具体的参数可以查看：[随机森林参数](https://scikit-learn.org/stable/modules/ensemble.html#random-forest-parameters)
>
><4>　模型训练
>```python
>model.fit(train_feature, train_label)
>```
>
><5>　模型保存
>```python
>joblib.dump(model, './model.pkl')
>```
>
><6>　模型调用
>```python
>##模型加载
>load_model = joblib.load('./model.pkl')
>
>##模型预测
>pred = load_model.predict_proba(test_feature)
>```

---

<p align='right'>Author : Junyu
<p align='right'>Date : 2020-04-23