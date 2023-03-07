# %% [markdown]
# # SVR
# ## Start with imports, downloading and preparing data.
from sklearn.svm import SVR
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import rsi
from ta.trend import sma_indicator

start_date = '2010-01-01'
end_date = '2020-01-01'
ticker = 'INVE-B.ST'
df = yf.download(ticker, start_date, end_date)
df.index = df.index.date
df.index.name = "Date"

df.head()

# %%
df = df.reset_index()
df = df.drop(['Date', 'Open', 'Low', 'Close', 'High'], axis=1)

# %% [markdown] m
# Create features:

df['5d_future_close'] = df['Adj Close'].shift(-5)
df['5d_close_future_pct'] = df['5d_future_close'].pct_change(5)
df['5d_close_pct'] = df['Adj Close'].pct_change(5)

# %%

feature_names = ['5d_close_pct']

for n in [50, 200]:  # Create the moving average indicator and divide by Adj_Close

    df['ma'+str(n)] = sma_indicator(df['Adj Close'],
                                    window=n, fillna=False) / df['Adj Close']
    df['rsi'+str(n)] = rsi(df['Adj Close'],
                           window=n, fillna=False)
    feature_names = feature_names + ['ma' + str(n), 'rsi' + str(n)]

# %%

# New features based on volume
new_features = ['Volume_1d_change', 'Volume_1d_change_SMA']
feature_names.extend(new_features)
df['Volume_1d_change'] = df['Volume'].pct_change()
df['Volume_1d_change_SMA'] = sma_indicator(
    df['Volume_1d_change'], window=5, fillna=False)

df = df.dropna()

# %%
# Create features and targets
# use feature_names for features; '5d_close_future_pct' for targets

features = df[feature_names]
targets = df['5d_close_future_pct']

# Create DataFrame from target column and feature columns
feature_and_target_cols = ['5d_close_future_pct']+feature_names
feat_targ_df = df[feature_and_target_cols]

# %%
scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(np.array(df).reshape(-1, 1))
df

# %%

training_size = int(len(df)*0.65)
test_size = len(df)-training_size
train_data, test_data = df[0:training_size, :], df[training_size:len(df), :1]
print("train_data: ", train_data.shape)
print("test_data: ", test_data.shape)
# %%


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]  # i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# %%


time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test", y_test.shape)
# %%


svr_rbf = SVR(kernel='rbf', C=1e2, gamma=0.1)
svr_rbf.fit(X_train, y_train)

# %%

train_predict = svr_rbf.predict(X_train)
test_predict = svr_rbf.predict(X_test)

train_predict = train_predict.reshape(-1, 1)
test_predict = test_predict.reshape(-1, 1)

print("Train data prediction:", train_predict.shape)
print("Test data prediction:", test_predict.shape)

# %%

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))
# %%

print("Train data RMSE: ", math.sqrt(mean_squared_error(original_ytrain,train_predict)))
print("Train data MSE: ", mean_squared_error(original_ytrain,train_predict))
print("Test data MAE: ", mean_absolute_error(original_ytrain,train_predict))
print("-------------------------------------------------------------------------------------")
print("Test data RMSE: ", math.sqrt(mean_squared_error(original_ytest,test_predict)))
print("Test data MSE: ", mean_squared_error(original_ytest,test_predict))
print("Test data MAE: ", mean_absolute_error(original_ytest,test_predict))
# %%
