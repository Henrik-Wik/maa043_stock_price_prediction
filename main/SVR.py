# %% [markdown]
# # SVR
# ## Start with imports, downloading and preparing data.

# [x] Import preprocessing and remove unnecessary imports

import matplotlib.pyplot as plt
from preprocessing import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

df = download_data()

# %%

[features, targets, features_target, feature_names] = create_features(df)

[X_train, X_test, y_train, y_test] = time_split(features, targets)

scaled_X_train, scaled_X_test, scaler = scale_data(
    X_train, X_test, y_train)
# %%

y_train = y_train.values.reshape(-1, 1)
scaler_pred = StandardScaler()
scaler_pred.fit(y_train)

# %%
svr_rbf = SVR(kernel='rbf', C=1e2, gamma=0.1)
svr_rbf.fit(scaled_X_train, y_train)
# %%

train_predict = svr_rbf.predict(scaled_X_train)
test_predict = svr_rbf.predict(scaled_X_test)

# %%

train_predict = scaler_pred.inverse_transform(train_predict.reshape(-1, 1))
test_predict = scaler_pred.inverse_transform(test_predict.reshape(-1, 1))

# %%
plt.figure(figsize=(8, 8), dpi=80)
plt.scatter(train_predict, y_train, label='train', s=5)
plt.scatter(test_predict, y_test, label='test', s=5)
plt.legend()
plt.show()

# %%
print("Train data R2: ", r2_score(
    y_train, train_predict))

print("Train data RMSE: ", mean_squared_error(
    y_train, train_predict, squared=False))
print("Train data MSE: ", mean_squared_error(y_train, train_predict))
print("Test data MAE: ", mean_absolute_error(y_train, train_predict))
print("-------------------------------------------------------------------------------------")
print("Test data R2: ", r2_score(
    y_test, test_predict))
print("Test data RMSE: ", mean_squared_error(
    y_test, test_predict, squared=False))
print("Test data MSE: ", mean_squared_error(y_test, test_predict))
print("Test data MAE: ", mean_absolute_error(y_test, test_predict))
# %%
