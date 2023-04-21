# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from ta.momentum import rsi
from ta.trend import sma_indicator

# %%

Stock = "SOBI.ST"

df = yf.download(Stock, "2010-01-01", "2020-01-01", period="1d")
df = df.reset_index()

# add EV to EBITDA from excel sheet

sheet_map = {
    "TELIA.ST": "TELIA",
    "HM-B.ST": "HM-B",
    "INVE-B.ST": "INVE-B",
    "VOLV-B.ST": "VOLV-B",
    "SOBI.ST": "SOBI",
}

sheet_name = sheet_map[Stock]
df2 = pd.read_excel("EV Ebitda.xlsx", sheet_name=sheet_name, index_col=0, header=0)

merged_df = pd.merge(df, df2, on="Date", how="outer")

df = merged_df.drop(["Date", "Open", "Low", "Close", "High"], axis=1)
df.fillna(method="ffill", inplace=True)

# %%

# Create features:
df["5d_future_close"] = df["Adj Close"].shift(-5)
df["5d_close_future_pct"] = df["5d_future_close"].pct_change(5)
df["5d_close_pct"] = df["Adj Close"].pct_change(5)

feature_names = []

for n in [
    14,
    30,
    50,
    100,
    200,
]:  # Create the moving average indicator and divide by Adj_Close
    df["ma" + str(n)] = (
        sma_indicator(df["Adj Close"], window=n, fillna=False) / df["Adj Close"]
    )
    df["rsi" + str(n)] = rsi(df["Adj Close"], window=n, fillna=False)
    feature_names = feature_names + ["ma" + str(n), "rsi" + str(n)]

# features based on volume
new_features = ["Volume_1d_change", "Volume_1d_change_SMA"]
feature_names.extend(new_features)
df["Volume_1d_change"] = df["Volume"].pct_change()
df["Volume_1d_change_SMA"] = sma_indicator(
    df["Volume_1d_change"], window=5, fillna=False
)

df.dropna(inplace=True)

# Create features and targets
# use feature_names for features; '5d_close_future_pct' for targets
features = df[feature_names]
targets = df["5d_close_future_pct"]

# Create DataFrame from target column and feature columns
feature_and_target_cols = ["5d_close_future_pct"] + feature_names
feat_targ_df = df[feature_and_target_cols]

# Uncomment to remove volume features

# features = features.drop(["Volume_1d_change", "Volume_1d_change_SMA"], axis=1)
# feat_targ_df = feat_targ_df.drop(
#     ["Volume_1d_change", "Volume_1d_change_SMA"], axis=1
# )
# feature_names = feature_names[:-2]

# %%

train_size = int(0.80 * targets.shape[0])
train_features = features[:train_size]
train_targets = targets[:train_size]
test_features = features[train_size:]
test_targets = targets[train_size:]

# %%

scaler = StandardScaler()
# transform using fit from training data.
scaled_train = scaler.fit_transform(train_features)
scaled_test = scaler.transform(test_features)

# used to inverse transform predicted data
pred_scaler = StandardScaler()
train = pred_scaler.fit(targets.values.reshape(-1, 1))


# %%

linear = LinearRegression()
linear.fit(train_features, train_targets)

linear_train_predict = linear.predict(train_features)
linear_test_predict = linear.predict(test_features)

# %%

svr = SVR(kernel="rbf", C=1e3, gamma=0.1)
svr.fit(train_features, train_targets)

svr_train_predict = svr.predict(train_features)
svr_test_predict = svr.predict(test_features)

# %%

rf = RandomForestRegressor(n_estimators=100, max_depth=10)
rf.fit(train_features, train_targets)

rf_train_predict = rf.predict(train_features)
rf_test_predict = rf.predict(test_features)

# %%

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(train_features, train_targets)

knn_train_predict = knn.predict(train_features)
knn_test_predict = knn.predict(test_features)

# %%

for model in [linear, svr, rf, knn]:
    train_predict = model.predict(train_features)
    test_predict = model.predict(test_features)

    train_r2 = r2_score(train_targets, train_predict)
    train_rmse = mean_squared_error(train_targets, train_predict, squared=False)
    train_mse = mean_squared_error(train_targets, train_predict)
    train_mae = mean_absolute_error(train_targets, train_predict)

    test_r2 = r2_score(test_targets, test_predict)
    test_rmse = mean_squared_error(test_targets, test_predict, squared=False)
    test_mse = mean_squared_error(test_targets, test_predict)
    test_mae = mean_absolute_error(test_targets, test_predict)

    print(f"{model} Train R2: ", train_r2)
    print(f"{model} Train RMSE: ", train_rmse)
    print(f"{model} Train MSE: ", train_mse)
    print(f"{model} Train MAE: ", train_mae)
    print(f"{model} Test R2: ", test_r2)
    print(f"{model} Test RMSE: ", test_rmse)
    print(f"{model} Test MSE: ", test_mse)
    print(f"{model} Test MAE: ", test_mae)
# %%

plt.figure(figsize=(8, 8), dpi=80)
plt.scatter(train_predict, train_targets, label="train", s=5)
plt.scatter(test_predict, test_targets, label="test", s=5)
plt.legend()
plt.show()
# %%
