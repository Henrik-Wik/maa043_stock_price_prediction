# %%
import pandas as pd
from models import *

# Run the tests

Stocks = {"^OMXSPI", "^OMX", "INVE-B.ST", "VOLV-B.ST", "TELIA.ST", "SOBI.ST", "HM-B.ST"}
models = [
    ("Linear", LinearRegression, optimize_linear),
    ("KNN", KNeighborsRegressor, optimize_knn),
    ("RF", RandomForestRegressor, optimize_rfr),
    ("SVR", SVR, optimize_svr),
    ("ANN", KerasRegressor, optimize_ann)
]

Latex_dict = {}

for Stock in Stocks:

    for model in models:
        Latex_dict[Stock] = model(Stock)


# %%

Latex_df = pd.DataFrame.from_dict(
    Latex_dict, orient="index", columns=["Linear", "KNN", "SVR", "RF", "ANN"]
)

# Latex_df = pd.DataFrame.from_dict(Latex_dict, orient="index", columns=["KNN"])

# %%
Transposed_df = Latex_df.transpose()

# %%

dfs = {}
for Stock in Stocks:
    dfs[Stock] = Transposed_df[Stock]

# %% export to txt
pd.set_option("display.max_colwidth", 1000)

for Stock in Stocks:
    dfs[Stock].to_string(f"../Data/Latex/{Stock}_df.txt", index=False)

# %%
