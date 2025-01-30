# -*- coding: utf-8 -*-
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
import loadData
import time
import os
import defines
from tensorflow import keras

tf.compat.v1.disable_eager_execution()

def model(input_tensor, hidden_size, output_size):
    with tf.variable_scope("Dense"):
        layer1 = tf.layers.dense(inputs=input_tensor,
                                 units=hidden_size,
                                 activation=tf.nn.relu,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.01))
        drop1 = tf.layers.dropout(inputs=layer1, rate=0.2)
        layer2 = tf.layers.dense(inputs=drop1,
                                 units=hidden_size,
                                 activation=tf.nn.relu,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.01))
        drop2 = tf.layers.dropout(inputs=layer2, rate=0.2)
        layer3 = tf.layers.dense(inputs=drop2,
                                 units=hidden_size,
                                 activation=tf.nn.relu,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.01))
        drop3 = tf.layers.dropout(inputs=layer3, rate=0.2)
        layer = tf.layers.dense(inputs=drop3, units=output_size)
    return layer

def create_model(input_size, hidden_size, output_size):
    model = keras.Sequential([
        keras.layers.Dense(input_shape=(input_size, ),
                           units=hidden_size,
                           activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(units=hidden_size,
                           activation=tf.nn.relu,
                           kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(output_size)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model

class LossHistory(keras.callbacks.Callback):
    def __init__(self, project_name):
        super(LossHistory, self).__init__()
        self.project_name = project_name

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        with open('loss_output.txt', 'a') as f:
            f.write(f'Project: {self.project_name}, Epoch {epoch + 1}: Loss = {loss}\n')

def train(rootPath, name, epoch):
    hidden_size = 256
    batch_size = 512
    train_num = 1000000

    path = os.path.join(rootPath, name)
    before_file = os.path.join(path, 'before_data.txt')
    after_file = os.path.join(path, 'after_data.txt')
    X, seq_len_X, max_x, min_x = loadData.before_func.get(name)(before_file)
    Y, seq_len_Y, max_y, min_y = loadData.after_func.get(name)(after_file)
    Y = np.array(Y)
    embed_X = X.shape[1]
    embed_Y = Y.shape[1]
    x_train = X[0:train_num]
    x_test = X[train_num:]
    y_train = Y[0:train_num]
    y_test = Y[train_num:]

    print(np.shape(x_train), np.shape(y_train))
    Model = create_model(input_size=embed_X, hidden_size=hidden_size, output_size=embed_Y)
    loss_history = LossHistory(project_name=name)  # 传递项目名称
    Model.fit(x_train, y_train, batch_size, epochs=epoch, callbacks=[loss_history])
    Model.evaluate(x_test, y_test, batch_size)
    Model.save(f'./model/{name}.hdf5')

if __name__ == '__main__':
    rootPath = defines.TM_PATH
    for i, name in enumerate(defines.TM):
        print(i + 1, name.value)
        train(rootPath, name.value, 3)
