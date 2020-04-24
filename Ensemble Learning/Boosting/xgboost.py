##载入所需库
import xgboost as xgb
import pandas as pd
import numpy as np

##载入训练数据
data = pd.read_csv('./train.csv').fillna(0)

##数据集拆分
label = data['pay_status']
feature = data.iloc[:, 3:]
train_rate = 0.7
random_index = np.random.permutation(feature.shape[0])
train_index, test_index = random_index[:int(train_rate * feature.shape[0])], \
                          random_index[int(train_rate * feature.shape[0]):]
train_label, test_label = label[train_index], label[test_index]
train_feature, test_feature = feature.iloc[train_index, :], feature.iloc[test_index, :]

##设置参数
params = {
    'booster':'gbtree',
    'min_child_weight': 0.8,
    'eta': 0.1,
    'colsample_bytree': 0.8,
    'max_depth': 8,
    'subsample': 0.8,
    'eval_metric': 'auc',
    'alpha': 0.1,
    'gamma': 0.1,
    'silent': 1,
    'objective': 'binary:logistic',
    'seed': 0
}

train_round = 150

##转化为DMatrix格式
dtrain = xgb.DMatrix(train_feature, label = train_label)
dtest = xgb.DMatrix(test_feature, label = test_label)

##模型训练
xgb_model = xgb.train(params, dtrain, num_boost_round = train_round, evals = [(dtest, 'test'),(dtrain, 'train')])

##模型保存
xgb_model.save_model('./test.model')

##模型调用
gbm = xgb.Booster(model_file = './test.model')

##模型预测
pre = gbm.predict(xgb.DMatrix(test_feature))
