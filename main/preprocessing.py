# %%

import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from ta.momentum import rsi
from ta.trend import sma_indicator


def download_data(ticker, start_date, end_date):

    df = yf.download(ticker, start_date, end_date)
    df = df.reset_index()
    df = df.drop(['Date', 'Open', 'Low', 'Close', 'High'], axis=1)

    return df


def create_features(df):

    # Create features:
    df['5d_future_close'] = df['Adj Close'].shift(-5)
    df['5d_close_future_pct'] = df['5d_future_close'].pct_change(5)
    df['5d_close_pct'] = df['Adj Close'].pct_change(5)

    feature_names = ['5d_close_pct']

    for n in [14, 30, 50, 200]:  # Create the moving average indicator and divide by Adj_Close

        df['ma'+str(n)] = sma_indicator(df['Adj Close'],
                                        window=n, fillna=False) / df['Adj Close']
        df['rsi'+str(n)] = rsi(df['Adj Close'],
                               window=n, fillna=False)
        feature_names = feature_names + ['ma' + str(n), 'rsi' + str(n)]

    # New features based on volume
    new_features = ['Volume_1d_change', 'Volume_1d_change_SMA']
    feature_names.extend(new_features)
    df['Volume_1d_change'] = df['Volume'].pct_change()
    df['Volume_1d_change_SMA'] = sma_indicator(
        df['Volume_1d_change'], window=5, fillna=False)

    df = df.dropna()

    # Create features and targets
    # use feature_names for features; '5d_close_future_pct' for targets
    features = df[feature_names]
    targets = df['5d_close_future_pct']

    # Create DataFrame from target column and feature columns
    feature_and_target_cols = ['5d_close_future_pct']+feature_names
    feat_targ_df = df[feature_and_target_cols]

    return features, targets, feat_targ_df


def scale_data(X_train, X_test):  # Standardization with dataframe as output

    scaler = StandardScaler().set_output(transform="pandas")

    scaler.fit(X_train)
    scaled_X_train = scaler.transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    return scaled_X_train, scaled_X_test


def time_split(targets, features):

    train_size = int(0.85*targets.shape[0])
    train_features = features[:train_size]
    train_targets = targets[:train_size]
    test_features = features[train_size:]
    test_targets = targets[train_size:]

    return train_features, test_features, train_targets, test_targets

# %%
