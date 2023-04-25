# %%
import matplotlib.pyplot as plt
import pandas as pd
from models import optimize_knn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

Stock = "TELIA.ST"


def KNNTest(Stock):
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
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{Stock} KNN")
    plt.legend()
    plt.savefig("../Graphs/" + filename + ".png")
    plt.show()

    plt.figure(figsize=(8, 8), dpi=80)
    plt.scatter(train_targets, train_predict - train_targets, label="train", s=5)
    plt.scatter(test_targets, test_predict - test_targets, label="test", s=5)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.title(f"{Stock} - KNN Residuals")
    plt.xlabel("Actual Values")
    plt.ylabel("Residuals")
    plt.legend()
    plt.savefig("../Graphs/" + filename + "_Residuals" + ".png")
    plt.show()

    Latex = f"KNN & ${train_r2:.3f}$ & ${test_r2:.3f}$ & ${train_mae:.3f}$ & ${test_mae:.3f}$ & ${train_mse:.3f}$ & ${test_mse:.3f}$ & ${train_rmse:.3f}$ & ${test_rmse:.3f}$ \\\\"

    return Latex
