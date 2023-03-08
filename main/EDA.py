# %% [markdown]
# # Downloading and preparing stock data

import matplotlib.pyplot as plt
from preprocessing import download_data

df = download_data("INVE-B.ST", "2010-01-01", "2020-01-01")

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