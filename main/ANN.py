# %%

import time

import matplotlib.pyplot as plt
import models as md
import numpy as np
import pandas as pd
import preprocessing as pp
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from models import optimize_ann
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main_ann(Stock, folder):

    if "^" in Stock:
        import preprocessing_Index as pp
    else:
        import preprocessing as pp

    data = pp.download_data(Stock)
    features, targets = pp.create_features(data)
    train_features, test_features, train_targets, test_targets = pp.time_split(
        features, targets
    )
    scaled_train_features, scaled_test_features = pp.normalize_data(
        train_features, test_features
    )

    best_params = optimize_ann(train_features, train_targets)
    print("Best hyperparameters:", best_params)

    n_layers = best_params["n_layers"]
    n_neurons = best_params["n_neurons"]
    learning_rate = best_params["learning_rate"]
    activation = best_params["activation"]

    model = Sequential()
    for i in range(n_layers):
        model.add(Dense(n_neurons, activation=activation))
    model.add(Dense(1, activation=activation))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)

    start_time = time.time()

    model.fit(scaled_train_features, train_targets, epochs=50, batch_size=32, verbose=1)

    end_time = time.time()

    train_predict = model.predict(scaled_train_features)
    test_predict = model.predict(scaled_test_features)

    train_r2 = r2_score(train_targets, train_predict)
    test_r2 = r2_score(test_targets, test_predict)

    # Compute the metrics and store them in variables

    train_rmse = mean_squared_error(train_targets, train_predict, squared=False)
    train_mse = mean_squared_error(train_targets, train_predict)
    train_mae = mean_absolute_error(train_targets, train_predict)

    test_rmse = mean_squared_error(test_targets, test_predict, squared=False)
    test_mse = mean_squared_error(test_targets, test_predict)
    test_mae = mean_absolute_error(test_targets, test_predict)

    elapsed_time = end_time - start_time
    time_per_trial = elapsed_time / 50

    # Create a dictionary with the metric names and values
    results_dict = {
        "Stock": Stock,
        "Model": "Neural Network",
        "Best Parameters": best_params,
        "Train R2": train_r2,
        "Test R2": test_r2,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse,
        "Train MSE": train_mse,
        "Test MSE": test_mse,
        "Train MAE": train_mae,
        "Test MAE": test_mae,
        "Total Time": elapsed_time,
        "Time per Trial": time_per_trial,
    }

    # Load the dictionary into a DataFrame
    results_df = pd.DataFrame.from_dict(results_dict, orient="index", columns=["Value"])

    # Save the DataFrame to a csv file

    filename = f"{Stock}_Optimized_ANN_results"

    results_df.to_string(f"{folder}Data/{filename}.txt")

    print(results_df)
    # plot the results

    plt.figure(figsize=(8, 8), dpi=80)
    plt.scatter(train_predict, train_targets, label="train", s=5)
    plt.scatter(test_predict, test_targets, label="test", s=5)
    plt.xlabel("Predicted", fontsize=22)
    plt.ylabel("Actual", fontsize=22)
    plt.title(f"{Stock} - Neural Network", fontsize=26)
    plt.legend(fontsize=18, markerscale=3)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(f"{folder}Graphs/{filename}.png")
    plt.show()

    # plt.figure(figsize=(8, 8), dpi=80)
    # plt.scatter(
    #     train_targets,
    #     np.ravel(train_predict) - np.ravel(train_targets),
    #     label="train",
    #     s=5,
    # )
    # plt.scatter(
    #     test_targets, np.ravel(test_predict) - np.ravel(test_targets), label="test", s=5
    # )
    # plt.axhline(y=0, color="r", linestyle="--")
    # plt.title(f"{Stock} - Neural Network Residuals", fontsize=26)
    # plt.xlabel("Actual Values", fontsize=20)
    # plt.ylabel("Residuals", fontsize=20)
    # plt.legend()
    # plt.savefig("../Graphs/" + filename + "_Residuals" + ".png")
    # plt.show()

    Latex = f"ANN & ${train_r2:.3f}$ & ${test_r2:.3f}$ & ${train_mae:.3f}$ & ${test_mae:.3f}$ & ${train_mse:.3f}$ & ${test_mse:.3f}$ & ${train_rmse:.3f}$ & ${test_rmse:.3f}$ & ${time_per_trial:.3f}$ \\\\"

    return Latex
