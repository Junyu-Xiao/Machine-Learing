##载入所需库
import lightgbm as lgb
import pandas as pd
import numpy as np
import sys
sys.path.append('./Module')
import KS_value as ks

##载入训练数据
data = pd.read_csv('./data/train.csv').fillna(0)

##数据集拆分
label = data['pay_status']
feature = data.iloc[:, 3:]
train_rate = 0.7
random_index = np.random.permutation(feature.shape[0])
train_index, test_index = random_index[:int(train_rate * feature.shape[0])], \
                          random_index[int(train_rate * feature.shape[0]):]
train_label, test_label = label[train_index], label[test_index]
train_feature, test_feature = feature.iloc[train_index, :], feature.iloc[test_index, :]

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

##将数据转化未Dataset格式
dtrain = lgb.Dataset(train_feature, label = train_label)
dtest = lgb.Dataset(test_feature, label = test_label)
lgb_model = lgb.train(params, dtrain, train_round, valid_sets = dtest)

##效果验证
score = np.array(model.predict(test_feature))
print('KS值为：{:.4f}'.format(ks.KS_value(test_label, score, 1)))

##模型保存
lgb_model.save_model('./test.model')
print('训练结束, 保存完毕!') 

##载入测试数据
data = pd.read_csv('/home/junyu/PycharmProjects/M1_pay/data/2019_12_in.csv').fillna(0)

##载入模型
lgb_model = lgb.Booster(model_file = './test.model')

##提取客户号
cust_no = data['cust_no']

##提取特征
feature = np.array(data.iloc[:, 2:])

##输出分数
score = lgb_model.predict(feature)

##客户号匹配
score = pd.Series(score)
cust_score = pd.concat([cust_no, score], axis = 1).rename(columns = {0 : 'score'})
cust_score = cust_score.groupby(['cust_no'])['score'].mean()
cust_score = pd.DataFrame(cust_score) 
