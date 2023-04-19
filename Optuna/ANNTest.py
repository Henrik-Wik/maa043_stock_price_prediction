# %%

import preprocessing as pp
import models as md
from models import optimize_ann
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow import keras


data = pp.download_data()
features, targets, feat_targ_df, feature_names = pp.create_features(data)
train_features, test_features, train_targets, test_targets = pp.time_split(
    features, targets
)
scaled_train_features, scaled_test_features, pred_scaler = pp.scale_data(
    train_features, test_features, targets
)

best_params = optimize_ann(train_features, train_targets)
print("Best hyperparameters:", best_params)


# %%
