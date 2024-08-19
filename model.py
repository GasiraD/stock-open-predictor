import sys
import warnings
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm

sns.set()
tf.compat.v1.random.set_random_seed(1234)


company_name = "SAMYANG CORPORATION"


df = pd.read_csv('train.csv')
df = df[df['Company_Name'] == company_name]
df.dropna(inplace=True)

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

minmax = MinMaxScaler().fit(df[['Open']].astype('float32'))
df_log = minmax.transform(df[['Open']].astype('float32'))
df_log = pd.DataFrame(df_log)

# minmax = MinMaxScaler().fit(df.iloc[:, -9:-8].astype('float32')) # open index
# df_log = minmax.transform(df.iloc[:, -9:-8].astype('float32')) # open index
# df_log = pd.DataFrame(df_log)




simulation_size = 1 #10
num_layers = 1
size_layer = 32 #128
timestamp = 1 #5
epoch = 10 #300
dropout_rate = 0.8
test_size = 10
learning_rate = 0.01

df_train = df_log
df.shape, df_train.shape




class Model(tf.keras.Model):
    def __init__(self, num_layers, size_layer, dropout_rate):
        super(Model, self).__init__()
        self.lstm_layers = [tf.keras.layers.LSTM(size_layer, return_sequences=True, dropout=dropout_rate) for _ in range(num_layers)]
        self.dense = tf.keras.layers.Dense(1)
    
    def call(self, inputs, training=False):
        x = inputs
        for layer in self.lstm_layers:
            x = layer(x, training=training)
        return self.dense(x[:, -1, :])  # return the last output

def calculate_accuracy(real, predict):
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
    return percentage * 100

def anchor(signal, weight):
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer

def forecast():
    modelnn = Model(num_layers=num_layers, size_layer=size_layer, dropout_rate=dropout_rate)
    modelnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')

    date_ori = pd.to_datetime(df['Date']).tolist()

    pbar = tqdm(range(epoch), desc='train loop')
    for i in pbar:
        total_loss, total_acc = [], []
        for k in range(0, df_train.shape[0] - 1, timestamp):
            index = min(k + timestamp, df_train.shape[0] - 1)
            batch_x = np.expand_dims(df_train.iloc[k : index, :].values, axis=0)
            batch_y = df_train.iloc[k + 1 : index + 1, :].values

            with tf.GradientTape() as tape:
                logits = modelnn(batch_x, training=True)
                loss = tf.reduce_mean(tf.square(batch_y - logits))
            
            grads = tape.gradient(loss, modelnn.trainable_variables)
            modelnn.optimizer.apply_gradients(zip(grads, modelnn.trainable_variables))
            
            total_loss.append(loss.numpy())
            total_acc.append(calculate_accuracy(batch_y[:, 0], logits.numpy()[:, 0]))
        
        pbar.set_postfix(cost=np.mean(total_loss), acc=np.mean(total_acc))
    
    future_day = test_size
    output_predict = np.zeros((df_train.shape[0] + future_day, df_train.shape[1]))
    output_predict[0] = df_train.iloc[0]

    for k in range(0, df_train.shape[0] - timestamp, timestamp):
        batch_x = np.expand_dims(df_train.iloc[k : k + timestamp, :].values, axis=0)
        logits = modelnn(batch_x, training=False)
        output_predict[k + 1 : k + timestamp + 1] = logits.numpy()
    
    future_inputs = np.expand_dims(df_train.iloc[-timestamp:, :].values, axis=0)
    for i in range(future_day):
        logits = modelnn(future_inputs, training=False)
        output_predict[-future_day + i] = logits.numpy()[-1]
        future_inputs = np.roll(future_inputs, shift=-1, axis=1)
        future_inputs[0, -1, :] = logits.numpy()[-1]
    
    output_predict = minmax.inverse_transform(output_predict)
    deep_future = anchor(output_predict[:, 0], 0.4)
    
    return deep_future

results = []
for i in range(simulation_size):
    print('simulation %d' % (i + 1))
    results.append(forecast())

date_ori = pd.to_datetime(df['Date']).tolist()
for i in range(test_size):
    date_ori.append(date_ori[-1] + timedelta(days=1))
date_ori = pd.Series(date_ori).dt.strftime(date_format='%Y-%m-%d').tolist()

accepted_results = []
for r in results:
    if (np.array(r[-test_size:]) < np.min(df['Open'])).sum() == 0 and \
    (np.array(r[-test_size:]) > np.max(df['Open']) * 2).sum() == 0:
        accepted_results.append(r)

accuracies = [calculate_accuracy(df['Open'].values, r[:-test_size]) for r in accepted_results]

plt.figure(figsize=(15, 5))
for no, r in enumerate(accepted_results):
    plt.plot(r, label='forecast %d' % (no + 1))
plt.plot(df['Open'], label='true trend', c='black')
plt.legend()
plt.title('average accuracy: %.4f' % (np.mean(accuracies)))

x_range_future = np.arange(len(results[0]))
plt.xticks(x_range_future[::30], date_ori[::30])

plt.show()