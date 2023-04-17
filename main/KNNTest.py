#%%
import preprocessing as pp
import models as md
from models import optimize_knn
from sklearn.neighbors import KNeighborsRegressor


data = pp.download_data()
features, targets, feat_targ_df, feature_names = pp.create_features(data)
train_features, test_features, train_targets, test_targets = pp.time_split(
    features, targets)
scaled_train_features, scaled_test_features, pred_scaler = pp.scale_data(
    train_features, test_features, targets)

best_params = optimize_knn(scaled_train_features, train_targets)
print("Best hyperparameters:", best_params)

#%%

knn = KNeighborsRegressor(**best_params)
knn.fit(scaled_train_features, train_targets)

#%%

knn_results = md.evaluation(
    knn, scaled_train_features, scaled_test_features, train_targets, test_targets)

print("KNN Results:")
print(knn_results)

#%%
#plot the results

import matplotlib.pyplot as plt

def plot_predictions(true_values, predictions, title):
    plt.scatter(true_values, predictions, alpha=0.5)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.show()

train_predictions = knn.predict(scaled_train_features)
test_predictions = knn.predict(scaled_test_features)

plot_predictions(train_targets, train_predictions, "knn - Training Data")
plot_predictions(test_targets, test_predictions, "knn - Testing Data")

# %%
