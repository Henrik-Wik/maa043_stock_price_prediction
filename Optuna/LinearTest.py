# %%
import matplotlib.pyplot as plt
import pandas as pd
from models import optimize_linear
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

Stocks = {"^OMXN40", "^OMX", "INVE-B.ST", "VOLV-B.ST", "TELIA.ST", "SOBI.ST", "HM-B.ST"}

for Stock in Stocks:
    if "^" in Stock:
        import preprocessing_Index as pp
    else:
        import preprocessing as pp

    data = pp.download_data(Stock)
    features, targets, feat_targ_df, feature_names = pp.create_features(data, Stock)
    train_features, test_features, train_targets, test_targets = pp.time_split(
        features, targets
    )
    scaled_train_features, scaled_test_features, pred_scaler = pp.scale_data(
        train_features, test_features, targets
    )

    best_params = optimize_linear(scaled_train_features, train_targets)
    print("Best hyperparameters:", best_params)

    linear = LinearRegression(**best_params)
    linear.fit(train_features, train_targets)

    train_predict = linear.predict(train_features)
    test_predict = linear.predict(test_features)

    train_predict = pred_scaler.inverse_transform(train_predict.reshape(-1, 1))
    test_predict = pred_scaler.inverse_transform(test_predict.reshape(-1, 1))

    # Compute the metrics and store them in variables
    train_r2 = r2_score(train_targets, train_predict)
    train_rmse = mean_squared_error(train_targets, train_predict, squared=False)
    train_mse = mean_squared_error(train_targets, train_predict)
    train_mae = mean_absolute_error(train_targets, train_predict)

    test_r2 = r2_score(test_targets, test_predict)
    test_rmse = mean_squared_error(test_targets, test_predict, squared=False)
    test_mse = mean_squared_error(test_targets, test_predict)
    test_mae = mean_absolute_error(test_targets, test_predict)

    # Create a dictionary with the metric names and values
    results_dict = {
        "Stock": Stock,
        "Model": "Linear Regression",
        "Best Parameters": best_params,
        "Train R2": train_r2,
        "Test R2": test_r2,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse,
        "Train MSE": train_mse,
        "Test MSE": test_mse,
        "Train MAE": train_mae,
        "Test MAE": test_mae,
        "": f"& ${train_r2:.4f}$ & ${test_r2:.4f}$ & ${train_mae:.4f}$ & ${test_mae:.4f}$ & ${train_mse:.4f}$ & ${test_mse:.4f}$ & ${train_rmse:.4f}$ & ${test_rmse:.4f}$",
    }

    # Load the dictionary into a DataFrame
    results_df = pd.DataFrame.from_dict(results_dict, orient="index", columns=["Value"])

    # Save the DataFrame to a csv file

    filename = f"{Stock}_Optimized_Linear_results"

    results_df.to_string("../Data/" + filename + ".txt")

    print(results_df)
    # plot the results

    plt.figure(figsize=(8, 8), dpi=80)
    plt.scatter(train_predict, train_targets, label="train", s=5)
    plt.scatter(test_predict, test_targets, label="test", s=5)
    plt.legend()
    plt.savefig("../Graphs/" + filename + ".png")
    plt.show()
