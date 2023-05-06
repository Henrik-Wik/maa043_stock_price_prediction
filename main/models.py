import keras.backend as K
import matplotlib.pyplot as plt
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

# Using timeseries split for cross validation
tscv = TimeSeriesSplit(n_splits=5)

# Linear Regression


def linreg_optuna(trial, X_train, y_train):
    # hyperparameters
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])

    linreg = LinearRegression(fit_intercept=fit_intercept, copy_X=True)
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


def rfr_optuna(trial, X_train, y_train):
    # hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 10, 150)
    max_features = trial.suggest_int("max_features", 1, 5)
    max_depth = trial.suggest_int("max_depth", 1, 5)

    rfr = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=42,
    )
    rfr.fit(X_train, y_train)

    scores = cross_val_score(rfr, X_train, y_train, cv=tscv, n_jobs=-1, scoring="r2")

    return scores.mean()


def optimize_rfr(X_train, y_train, n_trials=50):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: rfr_optuna(trial, X_train, y_train), n_trials=n_trials)

    return study.best_params


# K-Nearest Neighbors


def knn_optuna(trial, X_train, y_train):
    # hyperparameters
    n_neighbors = trial.suggest_int("n_neighbors", 3, 100)
    # algorithm = trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree"])
    leaf_size = trial.suggest_int("leaf_size", 1, 100)

    knn = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        algorithm="auto",
        weights="distance",
        p=2,
        leaf_size=leaf_size,
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


def Model(Stock, model_info):
    model_name, model_constructor, model_optimizer = model_info

    if "^" in Stock:
        import preprocessing_Index as pp
    else:
        import preprocessing as pp

    data = pp.download_data(Stock)
    features, targets = pp.create_features(data)
    train_features, test_features, train_targets, test_targets = pp.time_split(
        features, targets
    )
    # scaled_train_features, scaled_test_features = pp.scale_data(
    #     train_features, test_features
    # )

    scaled_train_features, scaled_test_features = pp.normalize_data(
        train_features, test_features
    )

    best_params = optimize_knn(scaled_train_features, train_targets)
    print("Best hyperparameters:", best_params)

    knn = KNeighborsRegressor(**best_params)
    knn.fit(scaled_train_features, train_targets)

    train_predict = knn.predict(scaled_train_features)
    test_predict = knn.predict(scaled_test_features)

    train_r2 = r2_score(train_targets, train_predict)
    test_r2 = r2_score(test_targets, test_predict)

    # Compute the metrics and store them in variables
    train_rmse = mean_squared_error(train_targets, train_predict, squared=False)
    train_mse = mean_squared_error(train_targets, train_predict)
    train_mae = mean_absolute_error(train_targets, train_predict)

    test_rmse = mean_squared_error(test_targets, test_predict, squared=False)
    test_mse = mean_squared_error(test_targets, test_predict)
    test_mae = mean_absolute_error(test_targets, test_predict)

    # Create a dictionary with the metric names and values
    results_dict = {
        "Stock": Stock,
        "Model": "KNN",
        "Best Parameters": best_params,
        "Train R2": train_r2,
        "Test R2": test_r2,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse,
        "Train MSE": train_mse,
        "Test MSE": test_mse,
        "Train MAE": train_mae,
        "Test MAE": test_mae,
    }

    # Load the dictionary into a DataFrame
    results_df = pd.DataFrame.from_dict(results_dict, orient="index", columns=["Value"])

    # Save the DataFrame to a csv file

    filename = f"{Stock}_Optimized_KNN_results"

    results_df.to_string("../Data/" + filename + ".txt")

    print(results_df)
    # plot the results

    plt.figure(figsize=(8, 8), dpi=80)
    plt.scatter(train_predict, train_targets, label="train", s=5)
    plt.scatter(test_predict, test_targets, label="test", s=5)
    plt.xlabel("Predicted", fontsize=22)
    plt.ylabel("Actual", fontsize=22)
    plt.title(f"{Stock} - KNN", fontsize=26)
    plt.legend(fontsize=18, markerscale=3)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig("../Graphs/" + filename + ".png")
    plt.show()

    # plt.figure(figsize=(8, 8), dpi=80)
    # plt.scatter(train_targets, train_predict - train_targets, label="train", s=5)
    # plt.scatter(test_targets, test_predict - test_targets, label="test", s=5)
    # plt.axhline(y=0, color="r", linestyle="--")
    # plt.title(f"{Stock} - KNN Residuals", fontsize=26)
    # plt.xlabel("Actual Values", fontsize=20)
    # plt.ylabel("Residuals", fontsize=20)
    # plt.legend()
    # plt.savefig("../Graphs/" + filename + "_Residuals" + ".png")
    # plt.show()

    Latex = f"KNN & ${train_r2:.3f}$ & ${test_r2:.3f}$ & ${train_mae:.3f}$ & ${test_mae:.3f}$ & ${train_mse:.3f}$ & ${test_mse:.3f}$ & ${train_rmse:.3f}$ & ${test_rmse:.3f}$ \\\\"

    return Latex
