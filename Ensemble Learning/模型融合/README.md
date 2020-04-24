# 模型融合介绍

>***模型融合是将多种不同模型融合在一起，可以提升模型的鲁棒性，模型融合其实与Bagging的思路类似，是将一些强的学习器集成在一起，只不过第二层不再是用简单的投票法或者平均法，而是将这些学习器的输出作为第二轮的特征输入进行进一步的训练。模型融合的方法有很多，下文主要介绍两种：Stacking与Blending。***

---

## 1. Stacking介绍
>***Stacking的优点是可以充分利用数据集，缺点是非常耗资源。***

### 1.1 Stacking原理介绍
>Stacking从字面翻译是堆叠的意思，其具体思维也和堆叠类似。其具体的操作如下：
>
>１．选择$M$个初级学习器，以及１个次级学习器。
>２．将数据集分为训练集1与测试集1。
>３．将训练集1分$K$等分，给每一份编号：$1,2,...,k$。
>４．遍历每份编号的数据集，该编号的数据作为测试集，其余编号的数据集作为训练集。
>５．以第４步生成的训练集与测试集训练$M$个学习器，将预测出的结果，储存为该编号的预测集。
>６．重复以上步骤，我们可以得到$M$份和训练集规模一样大的预测集。
>７．将所有预测集合并，得到一个新的训练集2。
>８．使用训练集2训练次级学习器，得到最终的次级学习器。
>９．保持超参不变，使用训练集1训练$M$个初级学习器，得到最终的$M$个初级学习器。
>１０．测试集1先用$M$个最终得到的初级学习器进行预测，将预测结果合并，得到测试集2。
>１１．测试集2再使用最终的次级学习器进行预测，得到最终的结果。
>
>从上面步骤来看，如果我们选择了4个初级学习器，并且将训练集5等分，那么我么你的需要训练$(4+1)\times5$个模型，资源消耗非常大，因此大多数情况下还是不建议使用模型融合这样的方法。

### 1.2 示例代码
><1>　加载所需包及数据
>```python
>##载入所需包
>from sklearn.ensemble import RandomForestClassifier
>import lightgbm as lgb
>import pandas as pd
>import numpy as np
>import joblib
>from sklearn.model_selection import KFold
>from sklearn.linear_model import LogisticRegression
>
>##载入训练数据
>data = pd.read_csv('./PycharmProjects/M1_pay/data/train.csv').fillna(0)
>```
>
><2>　数据集划分
>```python
>label = data['pay_status']
>feature = data.iloc[:, 3:]
>train_rate = 0.7
>random_index = np.random.permutation(feature.shape[0])
>train_index, test_index = random_index[:int(train_rate * feature.shape[0])], \
> random_index[int(train_rate * feature.shape[0]):]
>train_label, test_label = label[train_index], label[test_index]
>train_feature, test_feature = feature.iloc[train_index, :], feature.iloc[test_index, :] 
>```
>
><3>　设置模型参数
>```python
>##设置lgb参数
>params = {'boosting': 'gbdt',
>'objective': 'binary',
>'metric': 'auc',
>'learning_rate': 0.1,
>'num_leaves': 34,
>'max_depth': -1,
>'subsample': 0.8,
>'colsample_bytree': 0.8}
>
>train_round = 200
>```
>
><4>　生成初级模型的预测集
>
>```python
>##生成rf与lgb预测集
>new_feature = feature.copy()
>new_feature.loc[:, 'rf'] = 0
>new_feature.loc[:, 'lgb'] = 0
>```
>
><5>　训练集的拆分与学习器的训练
>```python
>##训练集拆分成４份
>sfold = KFold(n_splits = 4, shuffle = False)
>
>##循环训练
>for index1, index2 in sfold.split(train_index):
>
>##划分数据集
>in_train_feature = feature.iloc[train_index[index1]]
>in_test_feature = feature.iloc[train_index[index2]]
>in_train_label = label[train_index[index1]]
>in_test_label = label[train_index[index2]]
>
>
>##randomforest训练
>rf_model = RandomForestClassifier(n_estimators = 50, verbose = 0)
>rf_model.fit(in_train_feature, in_train_label)
>rf_out = rf_model.predict_proba(in_test_feature)[:,1]
>new_feature.loc[train_index[index2], 'rf'] = rf_out
>
>##lgb训练
>lgb_train = lgb.Dataset(in_train_feature, label = in_train_label)
>lgb_model = lgb.train(params, lgb_train, train_round)
>lgb_out = lgb_model.predict(in_test_feature)
>new_feature.loc[train_index[index2], 'lgb'] = lgb_out
>
>##总体rf
>rf_model = RandomForestClassifier(n_estimators = 50, verbose = 0)
>rf_model.fit(train_feature, train_label)
>rf_out = rf_model.predict_proba(test_feature)[:,1]
>new_feature.loc[test_index, 'rf'] = rf_out
>joblib.dump(rf_model, './rf.pkl')
>
>##总体lgb
>lgb_train = lgb.Dataset(train_feature, label = train_label)
>lgb_model = lgb.train(params, lgb_train, train_round)
>lgb_out = lgb_model.predict(test_feature)
>new_feature.loc[test_index, 'lgb'] = lgb_out 
>lgb_model.save_model('./lgb.model')
>```
>
><6>　生成次级学习器的特征
>```python
>new_feature = new_feature[['rf','lgb']]
>new_train = new_feature.iloc[train_index, :] 
>new_test = new_feature.iloc[test_index, :]
>```
>
><7>　训练次级学习器
>```python
>lr_model = LogisticRegression()
>lr_model.fit(new_train, train_label)
>pred = lr_model.predict_proba(new_test)
>joblib.dump(lr_model, './lr.pkl')
>```

## 2. Blending介绍
>***Blending的优点是资源耗费较小，缺点是数据利用不充分。***

### 2.1 Blending原理
>Blending的基本流程如下：
>１．选择$M$个初级学习器，１个次级学习器。
>２．将数据集分成３份，分别为训练集１，训练集２，测试集。
>３．用训练集１训练初级学习器。
>４．用训练好的初级学习器预测训练集２以及测试集，得到新的训练特征。
>５．用训练集２的新特征训练次级学习器。
>６．用测试集预测最终结果。

### 2.2 示例代码
><1>　载入包与数据
>```python
>##载入所需包
>from sklearn.ensemble import RandomForestClassifier
>import lightgbm as lgb
>import pandas as pd
>import numpy as np
>import joblib
>from sklearn.model_selection import KFold
>from sklearn.linear_model import LogisticRegression
>
>##载入训练数据
>data = pd.read_csv('./PycharmProjects/M1_pay/data/train.csv').fillna(0)
>```
>
><2>　数据集划分
>```python
>##数据集拆分
>label = data['pay_status']
>feature = data.iloc[:, 3:]
>random_index = np.random.permutation(feature.shape[0])
>train_rate1 = 0.5
>train_rate2 = 0.3
>split1 = int(train_rate1 * feature.shape[0])
>split2 = int((train_rate1 + train_rate2) * feature.shape[0])
>train_index1, train_index2, test_index = random_index[:split1],\
>                     random_index[split1:split2],\
>                     random_index[split2:]
>train_label1, train_label2, test_label = label[train_index1], label[train_index2], label[test_index]
>train_feature1, train_feature2, test_feature = feature.iloc[train_index1,:],\
>                           feature.iloc[train_index2,:],\
>                           feature.iloc[test_index,:]
>```
>
><3>　设置模型参数
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
>train_round = 200
>```
>
><4>　初级学习器训练
>
>```python
>##生成rf与lgb预测集
>new_feature = feature.copy()
>new_feature.loc[:, 'rf'] = 0
>new_feature.loc[:, 'lgb'] = 0
>
>##训练rf
>rf_model = RandomForestClassifier(n_estimators = 50, verbose = 0)
>rf_model.fit(train_feature1, train_label1)
>rf_out = rf_model.predict_proba(train_feature2)[:,1]
>new_feature.loc[train_index2, 'rf'] = rf_out
>joblib.dump(rf_model, './rf.pkl') 
>
>##训练lgb
>lgb_train = lgb.Dataset(train_feature1, label = train_label1)
>lgb_model = lgb.train(params, lgb_train, train_round)
>lgb_out = lgb_model.predict(train_feature2) 
>new_feature.loc[train_index2, 'lgb'] = lgb_out
>lgb_model.save_model('./lgb.model')
>```
>
><5>　生成次级学习器特征
>```python
>##生成次级学习器的特征
>new_feature = new_feature[['rf','lgb']]
>new_train = new_feature.iloc[train_index2, :] 
>new_test = new_feature.iloc[test_index, :]
>```
>
><6>　次级学习器训练
>```python
>##次级学习器训练
>lr_model = LogisticRegression()
>lr_model.fit(new_train, train_label2)
>pred = lr_model.predict_proba(new_test)
>joblib.dump(lr_model, './lr.pkl')
>```

---

<p align='right'>Author : Junyu
<p align='right'>Date : 2020-04-23

