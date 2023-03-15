# %% [markdown]
# # Downloading and preparing stock data

import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from preprocessing import *
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# Preprocessing

df = download_data()

[features, targets, features_targets, feature_names] = create_features(df)

[X_train, X_test, y_train, y_test] = time_split(features, targets)

scaled_X_train, scaled_X_test, pred_scaler = scale_data(
    X_train, X_test, y_train.values.reshape(-1, 1))

# %% [markdown]
# ## plot standardization

f, ax = plt.subplots(nrows=2, ncols=1)
X_train['5d_close_pct'].hist(ax=ax[0])
ax[0].set_title('Histogram of 5d_close_pct')

scaled_X_train['5d_close_pct'].hist(ax=ax[1])
ax[1].set_title('Histogram of scaled 5d_close_pct')

plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.5)

plt.show()

# %% [markdown]
# ## Neural Network Model

model_1 = Sequential()
model_1.add(
    Dense(100, input_dim=scaled_X_train.shape[1], activation='relu'))
model_1.add(Dense(20, activation='relu'))
model_1.add(Dense(1, activation='linear'))
model_1.compile(optimizer='adam', loss='mse')
history = model_1.fit(scaled_X_train, y_train, epochs=25)
# %%

plt.figure(figsize=(8, 8), dpi=80)
plt.plot(history.history['loss'])
plt.title('loss:'+str(round(history.history['loss'][-1], 6)))
plt.show()

# %%

train_preds = model_1.predict(scaled_X_train)
test_preds = model_1.predict(scaled_X_test)

train_preds = train_preds.reshape(-1, 1)
test_preds = test_preds.reshape(-1, 1)

train_predict = pred_scaler.inverse_transform(train_preds)
test_predict = pred_scaler.inverse_transform(test_preds)

print(r2_score(y_train, train_preds))
print(r2_score(y_test, test_preds))
print(mean_squared_error(y_train, train_preds))
print(mean_squared_error(y_test, test_preds))
# %%

plt.figure(figsize=(8, 8), dpi=80)
plt.scatter(train_preds, y_train, label='train', s=5)
plt.scatter(test_preds, y_test, label='test', s=5)
plt.legend()
plt.show()

# %%
