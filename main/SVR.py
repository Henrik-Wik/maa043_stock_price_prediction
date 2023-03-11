# %% [markdown]
# # SVR
# ## Start with imports, downloading and preparing data.

# [ ] Import preprocessing and remove unnecessary imports

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from preprocessing import *
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from ta.momentum import rsi
from ta.trend import sma_indicator

df = download_data()

# %%

[features, targets, features_targets] = create_features(df)

[X_train, X_test, y_train, y_test] = time_split(features, targets)

# %%

scaled_X_train, scaled_X_test, scaler = scale_data(X_train, X_test)

# %%

svr_rbf = SVR(kernel='rbf', C=1e2, gamma=0.1)
svr_rbf.fit(X_train, y_train)

# %%

train_predict = svr_rbf.predict(X_train)
test_predict = svr_rbf.predict(X_test)

# %%
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

print("Train data RMSE: ", math.sqrt(
    mean_squared_error(original_ytrain, train_predict)))
print("Train data MSE: ", mean_squared_error(original_ytrain, train_predict))
print("Test data MAE: ", mean_absolute_error(original_ytrain, train_predict))
print("-------------------------------------------------------------------------------------")
print("Test data RMSE: ", math.sqrt(
    mean_squared_error(original_ytest, test_predict)))
print("Test data MSE: ", mean_squared_error(original_ytest, test_predict))
print("Test data MAE: ", mean_absolute_error(original_ytest, test_predict))
# %%
