##加载包
import tensorflow as tf

##模型结构
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation = tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation = tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation = tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])

model.compile(optimizer = tf.keras.optimizers.Adam(0.00005),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])

##模型训练
model.fit(x_train, y_train, epochs = 50)
