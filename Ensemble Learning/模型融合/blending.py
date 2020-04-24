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
random_index = np.random.permutation(feature.shape[0])
train_rate1 = 0.5
train_rate2 = 0.3
split1 = int(train_rate1 * feature.shape[0])
split2 = int((train_rate1 + train_rate2) * feature.shape[0])
train_index1, train_index2, test_index = random_index[:split1],\
                                         random_index[split1:split2],\
                                         random_index[split2:]
train_label1, train_label2, test_label = label[train_index1], label[train_index2], label[test_index]
train_feature1, train_feature2, test_feature = feature.iloc[train_index1,:],\
                                               feature.iloc[train_index2,:],\
                                               feature.iloc[test_index,:]

##lgb参数
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

##训练rf
rf_model = RandomForestClassifier(n_estimators = 50, verbose = 0)
rf_model.fit(train_feature1, train_label1)
rf_out = rf_model.predict_proba(train_feature2)[:,1]
new_feature.loc[train_index2, 'rf'] = rf_out
joblib.dump(rf_model, './rf.pkl') 

##训练lgb
lgb_train = lgb.Dataset(train_feature1, label = train_label1)
lgb_model = lgb.train(params, lgb_train, train_round)
lgb_out = lgb_model.predict(train_feature2) 
new_feature.loc[train_index2, 'lgb'] = lgb_out
lgb_model.save_model('./lgb.model')

##生成次级学习器的特征
new_feature = new_feature[['rf','lgb']]
new_train = new_feature.iloc[train_index2, :] 
new_test = new_feature.iloc[test_index, :]

##次级学习器训练
lr_model = LogisticRegression()
lr_model.fit(new_train, train_label2)
pred = lr_model.predict_proba(new_test)
joblib.dump(lr_model, './lr.pkl')

