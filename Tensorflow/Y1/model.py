##加载包
import tensorflow as tf

##模型结构
class MyModel(tf.keras.Model):
    
    def __init__(self):
        super().__init__() 
        self.d1 = tf.keras.layers.Dense(128, activation = 'tanh') 
        self.d2 = tf.keras.layers.Dense(64, activation = 'relu') 
        self.d3 = tf.keras.layers.Dense(128, activation = 'tanh') 
        self.d4 = tf.keras.layers.Dense(64, activation = 'relu') 
        self.drop1 = tf.keras.layers.Dropout(0.2)
        self.drop2 = tf.keras.layers.Dropout(0.2)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.out1 = tf.keras.layers.Dense(3, activation = 'sigmoid') 
        self.out2 = tf.keras.layers.Dense(3, activation = 'sigmoid') 
        
    def call(self, x1, x2, training): 
        o1 = self.d1(x1) 
        o2 = self.bn1(o1)
        o3 = self.drop1(o2, training = training)
        o4 = self.d2(o3)
        out1 = self.out1(o4)
        o5 = self.d3(tf.concat([o4, x2], axis = 1))
        o6 = self.bn2(o5)
        o7 = self.drop2(o6, training = training)
        o8 = self.d4(o7)
        out2 = self.out2(o8)
        return (out1, out2)

model = MyModel()

##配置损失函数及其优化器
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(0.00005)
train_loss1 = tf.keras.metrics.Mean(name='train_loss1')
train_accuracy1 = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy1')
test_loss1 = tf.keras.metrics.Mean(name='test_loss1')
test_accuracy1 = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy1')
train_loss2 = tf.keras.metrics.Mean(name='train_loss2')
train_accuracy2 = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy2')
test_loss2 = tf.keras.metrics.Mean(name='test_loss2')
test_accuracy2 = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy2')

##设置测试及训练步骤
@tf.function
def train_step(x1, x2, labels1, labels2, training = True): 
    with tf.GradientTape(persistent=True) as tape:
        predictions1 = model(x1, x2, training = training)[0]
        loss1 = loss_object(labels1, predictions1) 
    gradients1 = tape.gradient(loss1, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients1, model.trainable_variables))
    del tape 
    
    with tf.GradientTape(persistent=True) as tape:
        predictions2 = model(x1, x2, training = training)[1]
        loss2 = loss_object(labels2, predictions2) 
    gradients2 = tape.gradient(loss2, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients2, model.trainable_variables))
    del tape 
    
    train_loss1(loss1)
    train_accuracy1(labels1, predictions1)
    train_loss2(loss2)
    train_accuracy2(labels2, predictions2)

@tf.function
def test_step(x1, x2, labels1, labels2, training = False): 
    pred1, pred2 = model(x1, x2, training = training)
    loss1 = loss_object(labels1, pred1)
    loss2 = loss_object(labels2, pred2)
    
    test_loss1(loss1) 
    test_accuracy1(labels1, pred1)
    test_loss2(loss2) 
    test_accuracy2(labels2, pred2)

##设置训练过程
EPOCHS = 1000
for epoch in range(EPOCHS):
    # 在下一个epoch开始时，重置评估指标
    train_loss1.reset_states()
    train_accuracy1.reset_states()
    train_loss2.reset_states()
    train_accuracy2.reset_states()
    test_loss1.reset_states()
    test_accuracy1.reset_states()
    test_loss2.reset_states()
    test_accuracy2.reset_states()
    
    for x1, x2, label1, label2 in d_train:
        train_step(x1, x2, label1, label2) 
        
    for x1, x2, label1, label2 in d_test:
        test_step(x1, x2, label1, label2) 
        
    template = 'Epoch {}, Train_Loss1: {:.6f}, Train_Accuracy1: {:.2f}%, Train_Loss2: {:.6f}, Train_Accuracy2: {:.2f}%, Test_Loss1: {:.6f}, Test_Accuracy1: {:.2f}%, Test_Loss2: {:.6f}, Test_Accuracy2: {:.2f}%'
    print(template.format(epoch + 1,
                         train_loss1.result(),
                         train_accuracy1.result()*100,
                         train_loss2.result(),
                         train_accuracy2.result()*100,
                         test_loss1.result(),
                         test_accuracy1.result()*100,
                         test_loss2.result(),
                         test_accuracy2.result()*100)) 
