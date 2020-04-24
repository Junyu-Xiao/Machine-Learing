##载入所需包
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

##载入训练数据
data = pd.read_csv('./PycharmProjects/M1_pay/data/train.csv').fillna(0)

##数据集拆分
label = data['pay_status']
feature = data.iloc[:, 3:]
train_rate = 0.7
random_index = np.random.permutation(feature.shape[0])
train_index, test_index = random_index[:int(train_rate * feature.shape[0])], \
                          random_index[int(train_rate * feature.shape[0]):]
train_label, test_label = label[train_index], label[test_index]
train_feature, test_feature = feature.iloc[train_index, :], feature.iloc[test_index, :] 

##设置lgb参数
params = {'boosting': 'gbdt',
          'objective': 'binary',
          'metric': 'auc',
          'learning_rate': 0.1,
          'num_leaves': 34,
          'max_depth': -1,
          'subsample': 0.8,
          'colsample_bytree': 0.8}

train_round = 200

##生成rf与lgb预测集
new_feature = feature.copy()
new_feature.loc[:, 'rf'] = 0
new_feature.loc[:, 'lgb'] = 0

##训练集拆分成４份
sfold = KFold(n_splits = 4, shuffle = False)

##循环训练
for index1, index2 in sfold.split(train_index):
    
    ##划分数据集
    in_train_feature = feature.iloc[train_index[index1]]
    in_test_feature = feature.iloc[train_index[index2]]
    in_train_label = label[train_index[index1]]
    in_test_label = label[train_index[index2]]
    
    
    ##randomforest训练
    rf_model = RandomForestClassifier(n_estimators = 50, verbose = 0)
    rf_model.fit(in_train_feature, in_train_label)
    rf_out = rf_model.predict_proba(in_test_feature)[:,1]
    new_feature.loc[train_index[index2], 'rf'] = rf_out
    
    ##lgb训练
    lgb_train = lgb.Dataset(in_train_feature, label = in_train_label)
    lgb_model = lgb.train(params, lgb_train, train_round)
    lgb_out = lgb_model.predict(in_test_feature)
    new_feature.loc[train_index[index2], 'lgb'] = lgb_out
    
##总体rf
rf_model = RandomForestClassifier(n_estimators = 50, verbose = 0)
rf_model.fit(train_feature, train_label)
rf_out = rf_model.predict_proba(test_feature)[:,1]
new_feature.loc[test_index, 'rf'] = rf_out
joblib.dump(rf_model, './rf.pkl')

##总体lgb
lgb_train = lgb.Dataset(train_feature, label = train_label)
lgb_model = lgb.train(params, lgb_train, train_round)
lgb_out = lgb_model.predict(test_feature)
new_feature.loc[test_index, 'lgb'] = lgb_out 
lgb_model.save_model('./lgb.model')

##生成次级学习器的特征
new_feature = new_feature[['rf','lgb']]
new_train = new_feature.iloc[train_index, :] 
new_test = new_feature.iloc[test_index, :]

##次级学习器训练
lr_model = LogisticRegression()
lr_model.fit(new_train, train_label)
pred = lr_model.predict_proba(new_test)
joblib.dump(lr_model, './lr.pkl')
