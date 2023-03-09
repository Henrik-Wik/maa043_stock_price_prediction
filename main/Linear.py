# %% [markdown]
# # Downloading and preparing stock data

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from preprocessing import *

df = download_data("INVE-B.ST", "2010-01-01", "2020-01-01")

# %% [markdown]
# Calculate correlation matrix

features, targets, feat_targ_df = create_features(df)

corr = feat_targ_df.corr()
print(corr)

# %%
# plot SMAs together

df[['ma14', 'ma30', 'ma50', 'ma200']].plot(figsize=(8, 5))
plt.title("INVE-B Stock Price Normalized", fontsize=17)
plt.xlabel("Time", fontsize=14)
plt.ylabel("Price", fontsize=14)
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)

# %%

plt.figure(figsize=(8, 8), dpi=80)
sns.heatmap(corr, annot=True, annot_kws={"size": 10})
plt.yticks(rotation=0, size=12)
plt.xticks(rotation=90, size=12)  # fix ticklab
plt.tight_layout()  # fits plot area to the plot, "tightly"
plt.show()  # show the plot

# %%
plt.figure(figsize=(8, 8), dpi=80)
plt.scatter(df['5d_close_future_pct'], df['ma200'], s=3)
plt.xlabel("5d_close_future_pct")
plt.ylabel("Volume_1d_change_SMA")
plt.show()

# %% [markdown]
# # Linear Regression Model
# $$ \Large y=\beta_0 + \beta_1x

linear_features = sm.add_constant(features)

X_train, X_test, y_train, y_test = time_split(targets, linear_features)

print(linear_features.shape, X_train.shape, X_test.shape)

# %%

model = sm.OLS(y_train, X_train)
results = model.fit()
print(results.summary())
print(results.pvalues)

train_predictions = results.predict(X_train)
test_predictions = results.predict(X_test)

# %%

plt.figure(figsize=(8, 8))
plt.scatter(train_predictions, y_train,
            alpha=0.2, color='b', label='train', s=6)
plt.scatter(test_predictions, y_test,
            alpha=0.2, color='r', label='test', s=6)

xmin, xmax = plt.xlim()
plt.plot(np.arange(xmin, xmax, 0.01), np.arange(xmin, xmax, 0.01), c='k')
plt.xlabel('Predictions')
plt.ylabel('Actual')
plt.legend()
# plt.xlim([-0.02,0.02])
# plt.ylim([-0.15,0.15])
