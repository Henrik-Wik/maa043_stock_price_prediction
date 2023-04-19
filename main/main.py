# %%
import models as md
import preprocessing as pp

# Preprocessing
data = pp.download_data()
features, targets, feat_targ_df, feature_names = pp.create_features(data)
train_features, test_features, train_targets, test_targets = pp.time_split(
    features, targets)
scaled_train_features, scaled_test_features, pred_scaler = pp.scale_data(
    train_features, test_features, targets)

#%%
# models with best parameters


# %%
# Training models

linreg = md.linear_regression(scaled_train_features, train_targets)
knn = md.knn_regression(scaled_train_features, train_targets)
ann = md.neural_network_regression(scaled_train_features, train_targets)
rf = md.random_forest_regression(scaled_train_features, train_targets)
svr = md.svr_optuna(scaled_train_features, train_targets)

# %%
# Results

linreg_results = md.evaluation(
    linreg, scaled_train_features, scaled_test_features, train_targets, test_targets)
knn_results = md.evaluation(
    knn, scaled_train_features, scaled_test_features, train_targets, test_targets)
ann_results = md.evaluation(
    ann, scaled_train_features, scaled_test_features, train_targets, test_targets)
rf_results = md.evaluation(rf, scaled_train_features,
                           scaled_test_features, train_targets, test_targets)
# svr_results = md.evaluation(
    # svr, scaled_train_features, scaled_test_features, train_targets, test_targets)

print("Linear Regression Results:")
print(linreg_results)
print("KNN Results:")
print(knn_results)
print("Neural Network Results:")
print(ann_results)
print("Random Forest Results:")
print(rf_results)
print("SVR Results:")
# print(svr_results)
