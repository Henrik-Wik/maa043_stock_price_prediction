# %%
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ta.momentum import rsi
from ta.trend import sma_indicator


def download_data(Stock):
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

    return df


def create_features(df, Stock):
    # Create features:
    df["5d_close_future"] = df["Adj Close"].shift(-5)
    # df["5d_close_future_pct"] = df["5d_close_future"].pct_change(5)
    # df["5d_close_pct"] = df["Adj Close"].pct_change(5)

    feature_names = ["Adj Close"]

    for n in [
        14,
        # 30,
        # 50,
        # 100,
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
    targets = df["5d_close_future"]

    # Create DataFrame from target column and feature columns
    feature_and_target_cols = ["5d_close_future"] + feature_names
    feat_targ_df = df[feature_and_target_cols]

    # Uncomment to remove volume features

    # features = features.drop(["Volume_1d_change", "Volume_1d_change_SMA"], axis=1)
    # feat_targ_df = feat_targ_df.drop(
    #     ["Volume_1d_change", "Volume_1d_change_SMA"], axis=1
    # )
    # feature_names = feature_names[:-2]

    return features, targets, feat_targ_df, feature_names


def time_split(features, targets):
    train_size = int(0.80 * targets.shape[0])
    train_features = features[:train_size]
    train_targets = targets[:train_size]
    test_features = features[train_size:]
    test_targets = targets[train_size:]

    return train_features, test_features, train_targets, test_targets


def scale_data(train, test, targets):  # Standardization with dataframe as output
    scaler = StandardScaler()
    # transform using fit from training data.
    scaled_train = scaler.fit_transform(train)
    scaled_test = scaler.transform(test)

    # used to inverse transform predicted data
    target_scaler = StandardScaler()
    target_scaler.fit(targets.values.reshape(-1, 1))

    return scaled_train, scaled_test, target_scaler


def normalize_data(X_train, X_test):  # normalization with dataframe as output
    scaler = MinMaxScaler(
        feature_range=(-1, 1),
    )

    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    return scaled_X_train, scaled_X_test, scaler
