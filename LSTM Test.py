# %% [markdown]
# # LSTM Stock Price Prediction

# %% [markdown]
# ## Start with imports, downloading and preparing data.
# %%
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
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
df1 = df.reset_index()['Adj Close']

plt.plot(df1)

# %% [markdown]
# We want to scale the dataset in order to use it for LSTM.

# %%

scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
df1

# %% [markdown]
# Now we can split it into training and testing

# %%
training_size = int(len(df1)*0.8)
test_size = len(df1)-training_size
train_data, test_data = df1[0:training_size,
                            :], df1[training_size:len(df1), :1]

print(training_size, test_size)

# %% [markdown]
# now we convert the values into a dataset matrix using this function.

# %%


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# %%
time_step = 100

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# %% [markdown]
# Now we reshape the input to be [sample, time steps, features]

# %%
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
print(X_train.shape, X_test.shape)

# %% [markdown]
# Now we can create the LSTM model

# %%
tf.keras.backend.clear_session()
model = Sequential()
model.add(LSTM(50, input_shape=(time_step, 1)))
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
