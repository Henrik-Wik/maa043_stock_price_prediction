# %%
import preprocessing as pp
import models as md
from models import optimize_knn
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


data = pp.download_data()
features, targets, feat_targ_df, feature_names = pp.create_features(data)
train_features, test_features, train_targets, test_targets = pp.time_split(
    features, targets
)
scaled_train_features, scaled_test_features, pred_scaler = pp.scale_data(
    train_features, test_features, targets
)

best_params = optimize_knn(scaled_train_features, train_targets)
print("Best hyperparameters:", best_params)

# %%

knn = KNeighborsRegressor(**best_params)
knn.fit(scaled_train_features, train_targets)

# %%

train_predict = knn.predict(scaled_train_features)
test_predict = knn.predict(scaled_test_features)

# %%

train_predict = pred_scaler.inverse_transform(train_predict.reshape(-1, 1))
test_predict = pred_scaler.inverse_transform(test_predict.reshape(-1, 1))


# %%

print("Train data R2: ", r2_score(train_targets, train_predict))

print(
    "Train data RMSE: ", mean_squared_error(train_targets, train_predict, squared=False)
)
print("Train data MSE: ", mean_squared_error(train_targets, train_predict))
print("Test data MAE: ", mean_absolute_error(train_targets, train_predict))
print(
    "-------------------------------------------------------------------------------------"
)
print("Test data R2: ", r2_score(test_targets, test_predict))
print("Test data RMSE: ", mean_squared_error(test_targets, test_predict, squared=False))
print("Test data MSE: ", mean_squared_error(test_targets, test_predict))
print("Test data MAE: ", mean_absolute_error(test_targets, test_predict))

# %%
# plot the results

plt.figure(figsize=(8, 8), dpi=80)
plt.scatter(train_predict, train_targets, label="train", s=5)
plt.scatter(test_predict, test_targets, label="test", s=5)
plt.legend()
plt.show()

# %%
