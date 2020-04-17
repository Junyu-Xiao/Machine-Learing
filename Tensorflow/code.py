##载入所需包　
import pandas as pd
import numpy as np
import tensorflow as tf

##载入数据
data = pd.read_csv('./PycharmProjects/M1_pay/data/train.csv').fillna(0)
data = data.iloc[:, 2:]

##标签提取、特征提取及其标准化
label = data['pay_status']
feature = data.iloc[:,1:]
feature = (feature - feature.mean())/feature.std()

##确定训练集、测试集索引
test_num = int(data.shape[0] * 0.2)
random_index = np.random.permutation(data.index)
test_index, train_index = random_index[:test_num], random_index[test_num:]

##确定测试集、训练集
test_label, train_label = np.array(label[test_index]).astype('int32'),\ 
                          np.array(label[train_index]).astype('int32')
test_feature, train_feature = np.array(feature.iloc[test_index,:]).astype('float32'),\
                              np.array(feature.iloc[train_index,:]).astype('float32')
dtrain = tf.data.Dataset.from_tensor_slices((train_feature, train_label)).batch(32)
dtest = tf.data.Dataset.from_tensor_slices((test_feature, test_label)).batch(32)

##设计模型结构
class Model(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(128, activation = 'relu')
        self.d2 = tf.keras.layers.Dense(256, activation = 'relu')
        self.d3 = tf.keras.layers.Dense(64, activation = 'relu')
        self.out = tf.keras.layers.Dense(2, activation = 'softmax')
        
    def call(self, feature):
        d1 = self.d1(feature)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        return self.out(d3)

model = Model() 

##设置损失函数以及梯度下降方法
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(0.00005)

##设置模型评估函数
train_loss = tf.keras.metrics.Mean(name='train_loss') 
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss') 
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

##设置模型训练步骤以及测试步骤
@tf.function
def train_step(feature, label):
    with tf.GradientTape(persistent=True) as tape:
        pred = model(feature)
        loss = loss_object(label, pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    del tape
    
    train_loss(loss)
    train_accuracy(label, pred)
    
@tf.function
def test_step(feature, label):
    pred = model(feature)
    loss = loss_object(label, pred)
    
    test_loss(loss)
    test_accuracy(label, pred)   

##模型训练
epoch = 50
for step in range(epoch):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states() 
    
    for feature, label in dtrain:
        train_step(feature, label)
    
    for feature, label in dtest:
        test_step(feature, label)
        
    out = '步数:{}, 训练误差:{:.6f}, 训练准确率:{:.2f}%, 测试误差:{:.6f}, 测试准确率:{:.2f}%'
    print(out.format(step+1,
                     train_loss.result(),
                     train_accuracy.result()*100,
                     test_loss.result(),
                     test_accuracy.result()*100))

##计算KS值
import sys 
sys.path.append('./Metric')
import KS_value as ks
test_pred = np.array(model(test_feature))[:,1]
ks_value = ks.KS_value(test_label, test_pred, 1)

##模型保存
model.save_weights('./test_weight')

##模型调用
model1 = Model()
model1.load_weights('./test_weight')
model1(test_feature)
