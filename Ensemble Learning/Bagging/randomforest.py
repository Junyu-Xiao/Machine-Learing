##载入所需包
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import joblib

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

##设置模型参数
model = RandomForestClassifier(n_estimators = 50, oob_score = True, verbose = 0)

##模型训练
model.fit(train_feature, train_label)

##模型保存
joblib.dump(model, './model.pkl')

##模型加载
load_model = joblib.load('./model.pkl')

##模型预测
pred = load_model.predict_proba(test_feature)
