# %%
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ta.momentum import rsi
from ta.trend import sma_indicator


Stock = "TELIA.ST"

df = yf.download(Stock, "2010-01-01", "2020-01-01", period="1d")
df = df.reset_index()

# add EV to EBITDA from excel sheet

if Stock == "TELIA.ST":
    df2 = pd.read_excel("EV Ebitda.xlsx", sheet_name="TELIA", index_col=0, header=0)

elif Stock == "HM-B.ST":
    df2 = pd.read_excel("EV Ebitda.xlsx", sheet_name="HM-B", index_col=0, header=0)

elif Stock == "INVE-B.ST":
    df2 = pd.read_excel("EV Ebitda.xlsx", sheet_name="INVE-B", index_col=0, header=0)

elif Stock == "VOLV-B.ST":
    df2 = pd.read_excel("EV Ebitda.xlsx", sheet_name="VOLV-B", index_col=0, header=0)

elif Stock == "SOBI.ST":
    df2 = pd.read_excel("EV Ebitda.xlsx", sheet_name="SOBI", index_col=0, header=0)

merged_df = pd.merge(df, df2, on="Date", how="outer")

df = merged_df.drop(["Date", "Open", "Low", "Close", "High"], axis=1)
df.fillna(method="ffill", inplace=True)

# Create features:
df["5d_future_close"] = df["Adj Close"].shift(-5)
df["5d_close_future_pct"] = df["5d_future_close"].pct_change(5)
df["5d_close_pct"] = df["Adj Close"].pct_change(5)

feature_names = ["5d_close_pct", "EVEBITDA"]

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

# features = features.drop(["Volume_1d_change", "Volume_1d_change_SMA"], axis=1)
# feat_targ_df = feat_targ_df.drop(
#     ["Volume_1d_change", "Volume_1d_change_SMA"], axis=1
# )
# feature_names = feature_names[:-2]
