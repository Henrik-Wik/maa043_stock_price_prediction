# %%
import matplotlib.pyplot as plt
import preprocessing as pp
from models import optimize_rfr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

Stock = "INVE-B.ST"

data = pp.download_data(Stock)
features, targets, feat_targ_df, feature_names = pp.create_features(data)
train_features, test_features, train_targets, test_targets = pp.time_split(
    features, targets
)
scaled_train_features, scaled_test_features, pred_scaler = pp.scale_data(
    train_features, test_features, targets
)

best_params = optimize_rfr(scaled_train_features, train_targets)
print("Best hyperparameters:", best_params)


# %%[markdown]
# ## Random Forest

rfr = RandomForestRegressor(**best_params)
rfr.fit(train_features, train_targets)

train_predict = rfr.predict(train_features)
test_predict = rfr.predict(test_features)

# %%

train_predict = pred_scaler.inverse_transform(train_predict.reshape(-1, 1))
test_predict = pred_scaler.inverse_transform(test_predict.reshape(-1, 1))

#%%

# Compute the metrics and store them in variables
train_r2 = r2_score(train_targets, train_predict)
train_rmse = mean_squared_error(train_targets, train_predict, squared=False)
train_mse = mean_squared_error(train_targets, train_predict)
train_mae = mean_absolute_error(train_targets, train_predict)

test_r2 = r2_score(test_targets, test_predict)
test_rmse = mean_squared_error(test_targets, test_predict, squared=False)
test_mse = mean_squared_error(test_targets, test_predict)
test_mae = mean_absolute_error(test_targets, test_predict)

# Create a dictionary with the metric names and values
results_dict = {
    "Stock": Stock,
    'Model': 'Random Forest',
    'Best Parameters': best_params,
    'Train R2': train_r2,
    'Train RMSE': train_rmse,
    'Train MSE': train_mse,
    'Train MAE': train_mae,
    'Test R2': test_r2,
    'Test RMSE': test_rmse,
    'Test MSE': test_mse,
    'Test MAE': test_mae,
}

# Load the dictionary into a DataFrame
results_df = pd.DataFrame.from_dict(results_dict, orient='index', columns=['Value'])

# Save the DataFrame to a csv file

filename = f"../Exports/{Stock}_Optimized_RF_results.csv"

results_df.to_csv(filename, index=False)

print(results_df)
# %%
# plot the results

plt.figure(figsize=(8, 8), dpi=80)
plt.scatter(train_predict, train_targets, label="train", s=5)
plt.scatter(test_predict, test_targets, label="test", s=5)
plt.legend()
plt.show()

# %%
