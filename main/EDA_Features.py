# %% [markdown]
# # Downloading and preparing stock data

import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import *

df = download_data()

# %%
# Stock price plot

df['Adj Close'].plot(figsize=(8, 5))
plt.title("INVE-B Stock Price", fontsize=17)
plt.xlabel("Time", fontsize=14)
plt.ylabel("Price", fontsize=14)
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)

# %%

plt.figure(figsize=(8, 5))
df['Adj Close'].pct_change().plot.hist(bins=50)

# %%
# Correlation coefficient

df['5d_future_close'] = df['Adj Close'].shift(-5)
df['5d_close_future_pct'] = df['5d_future_close'].pct_change(5)
df['5d_close_pct'] = df['Adj Close'].pct_change(5)

corr = df[['5d_close_pct', '5d_close_future_pct']].corr()
corr

# %%
# Scatterplot adj close vs future close

plt.figure(figsize=(8, 5))
plt.scatter(df['Adj Close'], df['5d_future_close'], s=3)
plt.xlabel("Adj Close")
plt.ylabel("5d_future_close")

# %% [markdown]
# scatterplot, 5d close future pct vs 5d close pct

plt.figure(figsize=(8, 5))
plt.scatter(df['5d_close_future_pct'], df['5d_close_pct'], s=3)
plt.xlabel("5d_close_future_pct")
plt.ylabel("5d_close_pct")

# %% [markdown]
# ## Feature Creation
# ### Calculate correlation matrix

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
# %%