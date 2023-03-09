# %% [markdown]
# # LSTM Stock Price Prediction
# ## Start with imports, downloading and preparing data.
# %%

# [ ] Import preprocessing and remove unnecessary imports
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import rsi
from ta.trend import sma_indicator
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

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

# %%
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
df = scaler.fit_transform(np.array(df))
df
# %% [markdown]
# We want to scale the dataset in order to use it for LSTM.

# %% [markdown]
# Now we can split it into training and testing

# %%
training_size = int(len(df)*0.8)
test_size = len(df)-training_size
train_data, test_data = df[0:training_size,
                           :], df[training_size:len(df), :1]

print(training_size, test_size)

# %% [markdown]
# now we convert the values into a dataset matrix using this function.

# %%


def create_dataset(dataset, time_step=1, num_features=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), :]
        dataX.append(a)
        dataY.append(dataset[i + time_step, :])
    return np.array(dataX), np.array(dataY)


# %%
time_step = 100

X_train, y_train = create_dataset(train_data, time_step, 11)
X_test, y_test = create_dataset(test_data, time_step, 11)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# %% [markdown]
# Now we reshape the input to be [sample, time steps, features]

# %%
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 11))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(X_train.shape, X_test.shape)

# %% [markdown]
# Now we can create the LSTM model

# %%
tf.keras.backend.clear_session()
model = Sequential()
model.add(LSTM(50, input_shape=(time_step, 11)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# %%
model.summary()

# %%
model.fit(X_train, y_train, validation_data=(
    X_test, y_test), epochs=100, batch_size=64, verbose=1)

# %%
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

print(train_predict.shape, test_predict.shape)

# %%
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# %%

print(math.sqrt(mean_squared_error(y_train, train_predict)))
print(math.sqrt(mean_squared_error(y_test, test_predict)))
print(r2_score(y_train, train_predict))
print(r2_score(y_test, test_predict))

# %%
look_back = 100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2) +
                1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# %%
