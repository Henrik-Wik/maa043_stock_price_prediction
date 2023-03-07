# %% [markdown]
# # Downloading and preparing stock data

from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import tensorflow as tf
import yfinance as yf
from keras.layers import Dense
from keras.models import Sequential
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler, scale
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
# check for missing values

df.isna().any()

# %% [markdown]
# ## Standardizing the data

scaled_train_features = scale(train_features)
scaled_test_features = scale(test_features)

f, ax = plt.subplots(nrows=2, ncols=1)
train_features.iloc[:, 2].hist(ax=ax[0])
ax[1].hist(scaled_train_features[:, 2])
plt.show()

# %%[markdown]
# ## K-NN

for n in range(2, 13, 1):
    # Create and fit the KNN model
    knn = KNeighborsRegressor(n_neighbors=n)

    # Fit the model to the training data
    knn.fit(scaled_train_features, train_targets)

    # Print number of neighbors and the score to find the best value of n
    print("n_neighbors =", n)
    print('train, test scores')
    print(knn.score(scaled_train_features, train_targets))
    print(knn.score(scaled_test_features, test_targets))
    print()  # prints a blank line

# %% [markdown]
# Evaluate model performance

# Create the model with the best-performing n_neighbors of 12
knn = KNeighborsRegressor(12)


# Fit the model
knn.fit(scaled_train_features, train_targets)

# Get predictions for train and test sets
train_predictions = knn.predict(scaled_train_features)
test_predictions = knn.predict(scaled_test_features)

# Plot the actual vs predicted values
plt.figure(figsize=(8, 8), dpi=80)
plt.scatter(train_predictions, train_targets, label='train', s=5)
plt.scatter(test_predictions, test_targets, label='test', s=5)
plt.legend()
plt.show()