# %% [markdown]
# # Downloading and preparing stock data

import matplotlib.pyplot as plt
import numpy as np
from preprocessing import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score

Stock = "INVE-B.ST"


df = download_data(Stock)

X, y, X_y, feature_names = create_features(df, Stock)

X_train, X_test, y_train, y_test = time_split(X, y)

# %%[markdown]
# ## Random Forest

# %%

rfr = RandomForestRegressor(n_estimators=200)
rfr.fit(X_train, y_train)

print(rfr.score(X_train, y_train))
print(rfr.score(X_test, y_test))

grid = {
    "n_estimators": [200],
    "max_depth": [3],
    "max_features": [4, 8],
    "random_state": [42],
}
test_scores = []

for g in ParameterGrid(grid):
    rfr.set_params(**g)  # ** is "unpacking" the dictionary
    rfr.fit(X_train, y_train)
    test_scores.append(rfr.score(X_test, y_test))


best_idx = np.argmax(test_scores)
print(test_scores[best_idx], ParameterGrid(grid)[best_idx])
# %%

rfr = RandomForestRegressor(
    n_estimators=200, max_depth=3, max_features=8, random_state=42
)
rfr.fit(X_train, y_train)

train_predictions = rfr.predict(X_train)
test_predictions = rfr.predict(X_test)

plt.figure(figsize=(8, 8), dpi=80)
plt.scatter(y_train, train_predictions, label="train", s=5)
plt.scatter(y_test, test_predictions, label="test", s=5)
plt.legend()
plt.show()


# %%

importances = rfr.feature_importances_

# Get the index of importances from greatest importance to least

sorted_index = np.argsort(importances)[::-1]
x = range(len(importances))

# Create tick labels

labels = np.array(feature_names)[sorted_index]
plt.figure(figsize=(8, 8), dpi=80)
plt.tick_params(axis="x", rotation=90)
plt.bar(x, importances[sorted_index], tick_label=labels)

# %%

r2_score_train = r2_score(y_train, train_predictions)
r2_score_test = r2_score(y_test, test_predictions)

print(r2_score_train, r2_score_test)
