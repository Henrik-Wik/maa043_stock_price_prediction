# %%
import matplotlib.pyplot as plt
import models as md
import pandas as pd
import preprocessing as pp
from models import optimize_linear
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

import optuna

Stock = "INVE-B.ST"

data = pp.download_data(Stock)
features, targets, feat_targ_df, feature_names = pp.create_features(data, Stock)
train_features, test_features, train_targets, test_targets = pp.time_split(
    features, targets
)
scaled_train_features, scaled_test_features, pred_scaler = pp.scale_data(
    train_features, test_features, targets
)


def objective(trial, X, y):
    name = trial.suggest_categorical(
        "Regressor", ["SVR", "RandomForest", "LinearRegression", "KNN"]
    )
    if name == "SVR":
        C = trial.suggest_float("C", 1e-2, 1e2, log=True)
        gamma = trial.suggest_float("gamma", 1e-2, 1e2, log=True)
        kernel = trial.suggest_categorical("kernel", ["poly", "rbf"])

        regressor = SVR(C=C, gamma=gamma, kernel=kernel)

    elif name == "RandomForest":
        n_estimators = trial.suggest_int("n_estimators", 1, 150)
        max_features = trial.suggest_int("max_features", 1, 5)
        max_depth = trial.suggest_int("max_depth", 1, 3)

        regressor = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, max_features=max_features
        )
    elif name == "LinearRegression":
        fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
        copy_X = trial.suggest_categorical("copy_X", [True, False])

        regressor = LinearRegression(fit_intercept=fit_intercept, copy_X=copy_X)

    elif name == "KNN":
        n_neighbors = trial.suggest_int("n_neighbors", 1, 20)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        algorithm = trial.suggest_categorical(
            "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
        )
        leaf_size = trial.suggest_int("leaf_size", 1, 20)
        p = trial.suggest_int("p", 1, 2)

        regressor = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
        )
    score = cross_val_score(regressor, X, y, n_jobs=-1, cv=3)
    accuracy = score.mean()

    return accuracy


if __name__ == "__main__":
    n_trials = 100
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, scaled_train_features, train_targets),
        n_trials=n_trials,
    )
    print(study.best_params)
