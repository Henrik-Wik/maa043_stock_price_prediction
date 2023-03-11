# %% [markdown]
# # LSTM Stock Price Prediction
# ## Start with imports, downloading and preparing data.
# [ ] Import preprocessing and remove unnecessary imports

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import LSTM, Dense
from keras.models import Sequential
from preprocessing import *
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, scale

df = download_data()

# %% [markdown] m
# Create features:

X, y, X_y = create_features(df)

# %%
# Split into training and testing sets

X_train, X_test, y_train, y_test = time_split(X, y)

# %%
# Normalize

# scaled_X_train, scaled_X_test = scale_data(X_train, X_test)


scaler = MinMaxScaler(feature_range=(-1, 1),
                      ).set_output(transform="pandas")

scaler.fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)

# %% [markdown]
# Now we reshape the input to be [sample, time steps, features]

# %%
timesteps = 1

scaled_X_train = scaled_X_train.to_numpy()
scaled_X_test = scaled_X_test.to_numpy()

# %%

scaled_X_train = scaled_X_train.reshape(-1, timesteps, scaled_X_train.shape[1])
scaled_X_test = scaled_X_test.reshape(-1, timesteps, scaled_X_test.shape[1])
print(scaled_X_train.shape, scaled_X_test.shape)

# %% [markdown]
# Now we can create the LSTM model

# %%
tf.keras.backend.clear_session()
model = Sequential()
model.add(LSTM(50, input_shape=(timesteps, 11)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# %%
model.summary()

# %%
model.fit(scaled_X_train, y_train, validation_data=(
    scaled_X_test, y_test), epochs=100, batch_size=64, verbose=1)

# %%
train_predict = model.predict(scaled_X_train)
test_predict = model.predict(scaled_X_test)

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
look_back = 1
trainPredictPlot = np.empty_like(df)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2) +
                1:len(df)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# %%
