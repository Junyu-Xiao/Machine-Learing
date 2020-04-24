# Boosting介绍

> ***Boosting是集成学习中最常用的算法之一，其为一个串联模型，即会根据前一轮训练得到的模型结构，去调整下一轮模型训练的样本分布，加大拟合效果差的样本权重。因为Boosting是串联模型，所以无法实现并行运算，需要在前一轮训练完成之后再进行下一轮。常用的Boosting模型有adaboost、xgboost、lightgbm等***

---

## 1.　XGBoost介绍
>XGBoost是最常用的Boosting模型，对传统的GBDT模型进行了改进，加入了二阶梯度提升了准确度，也加入了正则项以降低模型的偏差。

### 1.1　XGBoost原理
>XGBoost是一个经典的Boosting模型，其为一个加性模型，形式如下：
>$$
>\hat y_i=\sum_{k=1}^Nf_k(x_i)\\f_k\in F\\F:{f(x)=w_q(x)}，(q:R^m\to T，w\in R^T)
>$$
>其中$q$是树结构，$T$是叶节点数，$w$是对应叶节点。
>
>Boosting的每一轮的迭代都是为了使得模型拟合得更好，即降低损失函数的值，定义xgboost的目标函数为：
>$$
>L^{(t)}=\sum_{i=1}^nl(y_i,\hat y_i^{(t-1)}+f_t(x_i))+\Omega(f_t)\\ \Omega(f_t)=\gamma T+ \frac {1}{2}\lambda||w||^2
>$$
>
>对目标函数进行二街泰勒展开：
>$$
>L^{(t)}\approx \sum_{i=1}^n[l(y_i,\hat y_i^{(t-1)})+g_if_t(x_i)+\frac {1}{2}h_if_t^2(x_i)]+\Omega(f_t)\\
>g_i=\frac {\partial l(y_i,\hat y_i^{(t-1)})}{\part \hat y_i^{(t-1)}}；h_i=\frac {\part^2 l(y_i,\hat y_i^{(t-1)})}{\part (\hat y_i^{(t-1)})^2}
>$$
>
>由于前$t-1$棵树的损失函数$l(y_i,\hat y_i^{(t-1)})$与第$t$轮的训练无关，因此最终的目标函数是：
>$$
>Obj=argmin_{ft}\sum_{i=1}^n[g_if_t(x_i)+\frac {1}{2}h_if_t^2(x_i)]+\Omega(f_t)
>$$
>
>对目标函数进行如下变换：
>$$
>Obj=\sum_{i=1}^n[g_if_t(x_i)+\frac {1}{2}h_if_t^2(x_i)]+\gamma T+\frac {1}{2}\lambda \sum_{j=1}^Tw_j^2\\
>=\sum_{j=1}^T[(\sum_{x_i \in R_j}w_j)+\frac {1}{2}(\sum_{x_i \in R_j}h_i)w_j^2]+\gamma T + \frac {1}{2}\lambda \sum_{j=1}^Tw_j^2\\
>=\sum_{j=1}^T[(\sum_{x_i \in R_j}g_i)w_j+\frac {1}{2}(\sum_{x_i \in R_j}h_i+\lambda)w_j^2]+\gamma T
>$$
>其中$R_j$是叶节点的区域。
>
>可以看出$Obj$函数中的不确定变量变为了$w_j$，即叶节点分数，对此我们可以求出最优的叶节点分数，如下所示：
>令$Obj'=0$，得到对应叶节点的最优解为：
>$$
>w_j^*=-\frac {\sum_{x_i \in R_j}g_i}{\sum_{x_i \in R_j}h_i+\lambda}
>$$
>将最优解带入$Obj$可以得到：
>$$
>Obj=-\frac {1}{2}\sum_{j=1}^T\frac {(\sum_{x_i \in R_j}g_i)^2}{\sum_{x_i \in R_j}h_i+\lambda}+\gamma T
>$$
>因为$Obj$只与前$t-1$轮训练的模型的梯度$g_i$与$h_i$有关，因此只需要函数有二阶导数，就可以作为xgboost的损失函数。
>
>从化简得到的$Obj$函数，我们可以得到节点分裂的标准：分裂节点过后的$Obj$只要小于分裂前的$Obj$即可分裂，且需要找到最有的分裂节点使得$Obj$的下降幅度最大，对此节点分裂的目标函数可以定义为：
>$$
>Obj_{split}=Obj_{before}-Obj_{after}\\
>=-\frac {1}{2}\frac {(\sum_{x_i \in R_j}g_i)^2}{\sum_{x_i \in R_j}h_i+\lambda}+\gamma T+\frac {1}{2}[\frac {(\sum_{x_i \in R_L}g_i)^2}{\sum_{x_i \in R_L}h_i+\lambda}+\frac {(\sum_{x_i \in R_R}g_i)^2}{\sum_{x_i \in R_R}h_i+\lambda}]-\gamma (T+1)\\
>=\frac {1}{2}[\frac {(\sum_{x_i \in R_L}g_i)^2}{\sum_{x_i \in R_L}h_i+\lambda}+\frac {(\sum_{x_i \in R_R}g_i)^2}{\sum_{x_i \in R_R}h_i+\lambda}-\frac {(\sum_{x_i \in R_j}g_i)^2}{\sum_{x_i \in R_j}h_i+\lambda}]-\gamma
>$$
>对于一个树结构，每一步只分裂一个节点，因此分裂后的叶节点数由$T$上升为$T+1$，除了分裂的节点，树的其他节点均不发生变化，因此只考虑分裂节点的$Obj$变化。其中$R_j$是分裂前的节点；$R_L$是分裂后的左节点；$R_R$是分裂后的右节点。
>
>对此XGBoost的基本原理已经讲完了，如果想要更加深入了解XGBoost，可以去看陈天奇大佬的xgboost论文。

### 1.2　示例代码
>xgboost的安装：
>```python
>pip install xgboost
>```
>
><1>　载入所需库及数据：
>
>```python
>##载入所需库
>import xgboost as xgb
>import pandas as pd
>import numpy as np
>
>##载入训练数据
>data = pd.read_csv('./train.csv').fillna(0)
>```
>
><2>　数据集拆分（拆分为训练集与测试集）：
>```python
>label = data['pay_status']
>feature = data.iloc[:, 3:]
>train_rate = 0.7
>random_index = np.random.permutation(feature.shape[0])
>train_index, test_index = random_index[:int(train_rate * feature.shape[0])], \
>                           random_index[int(train_rate * feature.shape[0]):]
>train_label, test_label = label[train_index], label[test_index]
>train_feature, test_feature = feature.iloc[train_index, :], feature.iloc[test_index, :]
>```
>
><3>　将数据转化为DMatrix格式：
>```python
>dtrain = xgb.DMatrix(train_feature, label = train_label)
>dtest = xgb.DMatrix(test_feature, label = test_label)
>```
>DMatrix格式是xgboost的专用格式。
>
><4>　设置xgboost的参数：
>```python
>##参数
>params = {
>'booster':'gbtree',
>'min_child_weight': 0.8,
>'eta': 0.1,
>'colsample_bytree': 0.8,
>'max_depth': 8,
>'subsample': 0.8,
>'eval_metric': 'auc',
>'alpha': 0.1,
>'gamma': 0.1,
>'silent': 1,
>'objective': 'binary:logistic',
>'seed': 0
>}
>
>##训练步数
>train_round = 150
>```
>具体的参数介绍可以查看：[xgboost参数介绍](https://xgboost.readthedocs.io/en/latest//parameter.html)
>
><5>　模型训练：
>```python
>xgb_model = xgb.train(params, dtrain, num_boost_round = train_round, evals = [(dtest, 'test'),(dtrain, 'train')])
>```
>
><6>　模型保存：
>```python
>xgb_model.save_model('./test.model')
>```
>
><7>　模型调用：
>```python
>gbm = xgb.Booster(model_file = './test.model')
>```
>
><8>　模型预测：
>```python
>pre = gbm.predict(xgb.DMatrix(test_feature))
>```

---

## 2.　LightGBM介绍
> LightGBM是一个由微软提供的训练框架，其并非专一的Boosting训练框架，也包含了RandomForest这样的Bagging算法，其主要特点就是速度快，内存占用低。

### 2.1　LightGBM原理
>由于Boosting方面与XGBoost类似，在此就不再冗余，其对于XGBoost的改进有两个方面，一个称之为$GOSS(Gradient-based One-side Sample)$；另一个称之为$EFB(Exclusive Feature Bundling)$，下文将详细介绍这两种处理方式。
>
>#### GOSS（Gradient-based One-side Sample）
>$GOSS$的作用就是减少在涉及到样本的梯度的计算中的样本量，以降低计算量。例如在计算节点分裂过程中，$Obj$函数的增益涉及到样本的梯度，$GOSS$采用的方法如下：
>
>> １．将数据依据梯度从大到小排序。
>>
>> ２．取前$a\times 100$%的数据作为训练数据。
>>
>> ３．对余下$(1-a)\times 100$%的数据，随机取$b\times 100$%作为余下$(1-a)\times 100$%的数据的代表，并乘上权重$\frac {1-a}{b}$（以保持总体数据量不变）。
>>
>> 具体的梯度增益公式如下：
>> $$
>> Obj_{after}=\frac{1}{n}(\frac{(\sum_{x_i \in A_L}g_i+\frac{1-a}{b}\sum_{x_i \in B_L}g_i)^2}{n_l^j(d)}+\frac{(\sum_{x_i \in A_R}g_i+\frac{1-a}{b}\sum_{x_i \in B_R}g_i)^2}{n_r^j(d)})
>> $$
>>
>> GOSS所定义的梯度增益与实际的梯度增益相差不大，实际的差异的同阶无穷小为：
>> $$
>> O(\frac{1}{n_l^j(d)}+\frac{1}{n_r^j(d)}+\frac{1}{\sqrt{n}})
>> $$
>
>
>
>#### EFB（Exclusive Feature Bundling）
>
>$EFB$是将互斥的特征进行合并，那么什么是互斥特征？举个例子，有两个特征$A$与$B$，$A$有取值时，$B$没有取值；$B$有取值时，$A$没有取值，这样两个特征就称之为互斥特征。由于这样的特性，可以将互斥特征进行合并，合并的方式如下：
>
>> １．将特征$A$与$B$的数据离散化，形成直方图数据中的一个个$bins$。
>>
>> ２．如果特征$A$的取值范围为$[0,10)$，特征$B$的取值范围是$[0,20)$。
>>
>> ３．合并之后的特征的数据为离散数据，取值范围为$[0,30)$，其中$[10,30)$对应原来$B$特征的$[0,20)$。
>
>shilidaima
>
>因为LightGBM采用了这两种特殊处理方式，所以LightGBM的计算速度会比XGBoost快很多。

---

### 2.2　示例代码
>LightGBM安装
>```python
>pip install lightgbm
>```
>
><1>　载入所需库与数据
>```python
>##载入所需库
>import lightgbm as lgb
>import pandas as pd
>import numpy as np
>import sys
>sys.path.append('./Module')
>import KS_value as ks
>
>##载入训练数据
>data = pd.read_csv('./data/train.csv').fillna(0)
>```
>
><2>　数据集划分
>```python
>label = data['pay_status']
>feature = data.iloc[:, 3:]
>train_rate = 0.7
>random_index = np.random.permutation(feature.shape[0])
>train_index, test_index = random_index[:int(train_rate * feature.shape[0])], \
>                      random_index[int(train_rate * feature.shape[0]):]
>train_label, test_label = label[train_index], label[test_index]
>train_feature, test_feature = feature.iloc[train_index, :], feature.iloc[test_index, :]
>```
>
><3>　将数据转化未Dataset格式
>```python
>dtrain = lgb.Dataset(train_feature, label = train_label)
>dtest = lgb.Dataset(test_feature, label = test_label)
>```
>Dataset是lightgbm的专用数据格式。
>
><4>　设置参数
>```python
>##lgb参数
>params = {'boosting': 'gbdt',
>'objective': 'binary',
>'metric': 'auc',
>'learning_rate': 0.1,
>'num_leaves': 34,
>'max_depth': -1,
>'subsample': 0.8,
>'colsample_bytree': 0.8}
>
>##训练步数
>train_round = 200
>```
>具体的lightgbm参数可以参考：[LightGBM参数说明](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
>
><5>　模型训练
>```python
>lgb_model = lgb.train(params, dtrain, train_round, valid_sets = dtest)
>```
>
><6>　效果验证
>```python
>score = np.array(model.predict(test_feature))
>print('KS值为：{:.4f}'.format(ks.KS_value(test_label, score, 1)))
>```
>
><7>　模型保存
>```python
>lgb_model.save_model('./test.model')
>print('训练结束, 保存完毕!') 
>```
>
><8>　模型调用
>```python
>##载入测试数据
>data = pd.read_csv('/home/junyu/PycharmProjects/M1_pay/data/2019_12_in.csv').fillna(0)
>
>##载入模型
>lgb_model = lgb.Booster(model_file = './test.model')
>
>##提取特征
>feature = np.array(data.iloc[:, 2:])
>
>##输出分数
>score = lgb_model.predict(feature)
>```

---

<p align='right'>Author : Junyu
<p align='right'>Date : 2020-04-22

