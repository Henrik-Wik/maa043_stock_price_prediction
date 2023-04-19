import optuna
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from preprocessing import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


def linear_regression(X_train, y_train):
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)

    return linreg


def random_forest_regression(X_train, y_train):
    rfr = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    rfr.fit(X_train, y_train)

    return rfr


def knn_regression(X_train, y_train):
    knn = KNeighborsRegressor(n_neighbors=12)
    knn.fit(X_train, y_train)

    return knn


def knn_optuna(trial, X_train, y_train):
    # hyperparameters
    n_neighbors = trial.suggest_int("n_neighbors", 1, 20)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    algorithm = trial.suggest_categorical(
        "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
    )
    leaf_size = trial.suggest_int("leaf_size", 1, 50)
    p = trial.suggest_int("p", 1, 2)

    knn = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
    )
    knn.fit(X_train, y_train)

    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring="r2")

    return np.mean(scores)


def optimize_knn(X_train, y_train, n_trials=100):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: knn_optuna(trial, X_train, y_train), n_trials=n_trials)

    best_params = optimize_knn(X_train, y_train)

    knn = KNeighborsRegressor(**best_params)
    knn.fit(X_train, y_train)

    return study.best_params, knn


def svr_optuna(trial, X_train, y_train):
    # hyperparameters
    C = trial.suggest_float("C", 1e-2, 1e2, log=True)
    gamma = trial.suggest_float("gamma", 1e-2, 1e2, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])

    svr = SVR(kernel=kernel, C=C, gamma=gamma)
    svr.fit(X_train, y_train)

    scores = cross_val_score(
        svr, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    )
    return np.mean(scores)


def optimize_svr(X_train, y_train, n_trials=100):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: svr_optuna(trial, X_train, y_train), n_trials=n_trials)

    return study.best_params




def neural_network_regression(X_train, y_train):
    model = Sequential()
    model.add(Dense(100, input_dim=X_train.shape[1], activation="relu"))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer="adam", loss="mse")

    return model


def evaluation(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    scaler_y=None,
    is_ann=False,
    inverse_transform=False,
):
    if is_ann:
        y_train_pred = model.predict(X_train).flatten()
        y_test_pred = model.predict(X_test).flatten()
    else:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

    if inverse_transform and scaler_y is not None:
        y_train = scaler_y.inverse_transform(y_train)
        y_test = scaler_y.inverse_transform(y_test)
        y_train_pred = scaler_y.inverse_transform(y_train_pred)
        y_test_pred = scaler_y.inverse_transform(y_test_pred)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)

    return {
        "r2_train": r2_train,
        "r2_test": r2_test,
        "mae_train": mae_train,
        "mae_test": mae_test,
        "mse_train": mse_train,
        "mse_test": mse_test,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
    }
