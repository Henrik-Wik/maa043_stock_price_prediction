#%%
import preprocessing as pp
import models as md
from models import optimize_svr
from sklearn.svm import SVR


data = pp.download_data()
features, targets, feat_targ_df, feature_names = pp.create_features(data)
train_features, test_features, train_targets, test_targets = pp.time_split(
    features, targets)
scaled_train_features, scaled_test_features, pred_scaler = pp.scale_data(
    train_features, test_features, targets)

best_params = optimize_svr(scaled_train_features, train_targets)
print("Best hyperparameters:", best_params)

#%%

svr = SVR(**best_params)
svr.fit(scaled_train_features, train_targets)

#%%

svr_results = md.evaluation(
    svr, scaled_train_features, scaled_test_features, train_targets, test_targets)

print("SVR Results:")
print(svr_results)

#%%
#plot the results

import matplotlib.pyplot as plt

def plot_predictions(true_values, predictions, title):
    plt.scatter(true_values, predictions, alpha=0.5)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.show()

train_predictions = svr.predict(scaled_train_features)
test_predictions = svr.predict(scaled_test_features)

plot_predictions(train_targets, train_predictions, "SVR - Training Data")
plot_predictions(test_targets, test_predictions, "SVR - Testing Data")

# %%
