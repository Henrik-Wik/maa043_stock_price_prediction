import keras.backend as K
import optuna
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from preprocessing import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

tscv = TimeSeriesSplit(n_splits=5)


# Linear Regression


def linear_regression(X_train, y_train):
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)

    return linreg


# %%


def linreg_optuna(trial, X_train, y_train):
    # hyperparameters
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
    copy_X = trial.suggest_categorical("copy_X", [True, False])

    linreg = LinearRegression(fit_intercept=fit_intercept, copy_X=copy_X)
    linreg.fit(X_train, y_train)

    scores = cross_val_score(linreg, X_train, y_train, cv=tscv, n_jobs=-1, scoring="r2")

    return scores.mean()


def optimize_linear(X_train, y_train, n_trials=50):
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: linreg_optuna(trial, X_train, y_train), n_trials=n_trials
    )

    return study.best_params


# Random Forest


def random_forest_regression(X_train, y_train):
    rfr = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    rfr.fit(X_train, y_train)

    return rfr


def rfr_optuna(trial, X_train, y_train):
    # hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 10, 150)
    max_features = trial.suggest_int("max_features", 1, 5)
    max_depth = trial.suggest_int("max_depth", 1, 5)

    rfr = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, max_features=max_features
    )
    rfr.fit(X_train, y_train)

    scores = cross_val_score(rfr, X_train, y_train, cv=tscv, n_jobs=-1, scoring="r2")

    return scores.mean()


def optimize_rfr(X_train, y_train, n_trials=50):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: rfr_optuna(trial, X_train, y_train), n_trials=n_trials)

    return study.best_params


# K-Nearest Neighbors


def knn_regression(X_train, y_train):
    knn = KNeighborsRegressor(n_neighbors=12)
    knn.fit(X_train, y_train)

    return knn


def knn_optuna(trial, X_train, y_train):
    # hyperparameters
    n_neighbors = trial.suggest_int("n_neighbors", 3, 30)
    leaf_size = trial.suggest_int("leaf_size", 1, 50)

    knn = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        algorithm="auto",
        leaf_size=leaf_size,
        weights="distance",
        p=2,
    )
    knn.fit(X_train, y_train)

    scores = cross_val_score(knn, X_train, y_train, cv=tscv, n_jobs=-1, scoring="r2")

    return scores.mean()


def optimize_knn(X_train, y_train, n_trials=50):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: knn_optuna(trial, X_train, y_train), n_trials=n_trials)

    return study.best_params


# Support Vector Regression


def svr_optuna(trial, X_train, y_train):
    # hyperparameters
    C = trial.suggest_float("C", 1, 100, log=True)
    gamma = trial.suggest_float("gamma", 1e-4, 1, log=True)
    kernel = trial.suggest_categorical("kernel", ["rbf"])

    svr = SVR(kernel=kernel, C=C, gamma=gamma)
    svr.fit(X_train, y_train)

    scores = cross_val_score(svr, X_train, y_train, cv=tscv, n_jobs=-1, scoring="r2")
    return scores.mean()


def optimize_svr(X_train, y_train, n_trials=50):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: svr_optuna(trial, X_train, y_train), n_trials=n_trials)

    return study.best_params


# Artificial Neural Network


def neural_network_regression(X_train, y_train):
    model = Sequential()
    model.add(Dense(100, input_dim=X_train.shape[1], activation="relu"))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer="adam", loss="mse")

    return model


def ann_model(trial):
    # hyperparameters
    n_layers = trial.suggest_int("n_layers", 1, 4)
    n_neurons = trial.suggest_int("n_neurons", 16, 256)
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1, log=True)
    activation = trial.suggest_categorical("activation", ["relu", "sigmoid"])

    # Define the architecture of the neural network
    model = Sequential()
    for i in range(n_layers):
        model.add(Dense(n_neurons, activation=activation))
    model.add(Dense(1, activation=activation))

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)

    return model


def ann_optuna(trial, X_train, y_train):
    K.clear_session()

    # Create the KerasRegressor model
    model = KerasRegressor(
        build_fn=lambda: ann_model(trial), epochs=10, batch_size=32, verbose=0
    )

    # Calculate cross-validation scores
    scores = []
    for train_index, test_index in tscv.split(X_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)
        score = r2_score(y_test_fold, y_pred)
        scores.append(score)

    return np.mean(scores)


def optimize_ann(X_train, y_train, n_trials=50):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: ann_optuna(trial, X_train, y_train), n_trials=n_trials)

    return study.best_params


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
