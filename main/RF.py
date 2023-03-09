# %% [markdown]
# # Downloading and preparing stock data

# [ ] Import preprocessing and remove unnecessary imports

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

# %%[markdown]
# ## Random Forest

# %%

rfr = RandomForestRegressor(n_estimators=200)
rfr.fit(train_features, train_targets)

print(rfr.score(train_features, train_targets))
print(rfr.score(test_features, test_targets))

grid = {'n_estimators': [200], 'max_depth': [3],
        'max_features': [4, 8], 'random_state': [42]}
test_scores = []

for g in ParameterGrid(grid):
    rfr.set_params(**g)  # ** is "unpacking" the dictionary
    rfr.fit(train_features, train_targets)
    test_scores.append(rfr.score(test_features, test_targets))


best_idx = np.argmax(test_scores)
print(test_scores[best_idx], ParameterGrid(grid)[best_idx])
# %%

rfr = RandomForestRegressor(
    n_estimators=200, max_depth=3, max_features=8, random_state=42)
rfr.fit(train_features, train_targets)

train_predictions = rfr.predict(train_features)
test_predictions = rfr.predict(test_features)

plt.figure(figsize=(8, 8), dpi=80)
plt.scatter(train_targets, train_predictions, label='train', s=5)
plt.scatter(test_targets, test_predictions, label='test', s=5)
plt.legend()
plt.show()


# %%

importances = rfr.feature_importances_

# Get the index of importances from greatest importance to least

sorted_index = np.argsort(importances)[::-1]
x = range(len(importances))

# Create tick labels


plt.figure(figsize=(8, 8), dpi=80)
plt.bar(x, importances[sorted_index])
