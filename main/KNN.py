# %% [markdown]
# # Downloading and preparing stock data

import matplotlib.pyplot as plt
from preprocessing import *
from sklearn.neighbors import KNeighborsRegressor

df = download_data()

# %%
# Feature creation
X, y, X_y_df, feature_names = create_features(df)

# %%
# Train test splitting

X_train, X_test, y_train, y_test = time_split(X, y)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# %% [markdown]
# ## Standardizing the data

scaled_X_train, scaled_X_test, scaler = scale_data(X_train, X_test)

# %%[markdown]
# ## K-NN

for n in range(2, 13, 1):
    # Create and fit the KNN model
    knn = KNeighborsRegressor(n_neighbors=n)

    # Fit the model to the training data
    knn.fit(scaled_X_train, y_train)

    # Print number of neighbors and the score to find the best value of n
    print("n_neighbors =", n)
    print('train, test scores')
    print(knn.score(scaled_X_train, y_train))
    print(knn.score(scaled_X_test, y_test))
    print()  # prints a blank line

# %% [markdown]
# Create the model with the best-performing n_neighbors of 12

knn = KNeighborsRegressor(12)

# Fit the model
knn.fit(scaled_X_train, y_train)

# Get predictions for train and test sets
train_predictions = knn.predict(scaled_X_train)
test_predictions = knn.predict(scaled_X_test)

# Plot the actual vs predicted values
plt.figure(figsize=(8, 8), dpi=80)
plt.scatter(train_predictions, y_train, label='train', s=5)
plt.scatter(test_predictions, y_test, label='test', s=5)
plt.legend()
plt.show()

# %%
