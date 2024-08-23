import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from math import sqrt



#hyperparameter
company_name = "SAMYANG CORPORATION"

num_layers = 1
size_layer = 32
epoch = 50
dropout_rate = 0.5
time_step = 30
batch_size = 16
train_size = 0.7


# 데이터 읽기 및 필터링
df = pd.read_csv('train.csv')
df = df[df['Company_Name'] == company_name]
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')
df.dropna(inplace=True)


# 테스트용 임시 수정
# df = df.head(365)


# 데이터 선택 및 정규화
open = df['Open'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
open_scaled = scaler.fit_transform(open)



# 테스트 데이터 분할
train_size = int(len(open_scaled) * train_size)
train_data, test_data = open_scaled[:train_size], open_scaled[train_size:]


# 시계열 데이터 준비

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)


# 데이터 형태 변경 (LSTM)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


# LSTM 모델 정의

model = tf.keras.models.Sequential()
for _ in range(num_layers):
    model.add(tf.keras.layers.LSTM(size_layer, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(tf.keras.layers.Dropout(dropout_rate))

model.add(tf.keras.layers.LSTM(size_layer))
model.add(tf.keras.layers.Dropout(dropout_rate))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_split=0.1)

# 모델 평가

## loss 평가
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

## rmse 평가
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
train_predictions = scaler.inverse_transform(train_predictions)
y_train_unscaled = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
test_predictions = scaler.inverse_transform(test_predictions)
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

def calculate_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    return rmse


train_rmse = calculate_rmse(y_train_unscaled, train_predictions.flatten())
test_rmse = calculate_rmse(y_test_unscaled, test_predictions.flatten())

print(f'Training RMSE: {train_rmse:.4f}')
print(f'Testing RMSE: {test_rmse:.4f}')


# 예측 및 결과 시각화
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(df['Date'].iloc[-len(y_test):], y_test, color='blue', label='Actual Prices')
plt.plot(df['Date'].iloc[-len(predictions):], predictions, color='red', label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{company_name} Price Prediction')
plt.legend()
plt.show()


