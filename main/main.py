# %%
import pandas as pd
from ANN import main_ann
from KNN import main_knn
from Linear import main_linear
from RF import main_rf
from SVR import main_svr

# Run the tests

folder = "../"

Stocks = {"^OMXSPI", "^OMX", "INVE-B.ST", "VOLV-B.ST", "TELIA.ST", "SOBI.ST", "HM-B.ST"}

Latex_dict = {}

for Stock in Stocks:
    Latex_dict[Stock] = (
        main_linear(Stock, folder),
        main_knn(Stock, folder),
        main_svr(Stock, folder),
        main_rf(Stock, folder),
        main_ann(Stock, folder),
    )


# %%

Latex_df = pd.DataFrame.from_dict(
    Latex_dict, orient="index", columns=["Linear", "KNN", "SVR", "RF", "ANN"]
)

# %%
Transposed_df = Latex_df.transpose()

# %%

dfs = {}
for Stock in Stocks:
    dfs[Stock] = Transposed_df[Stock]

# %% export to txt
pd.set_option("display.max_colwidth", 1000)

for Stock in Stocks:
    dfs[Stock].to_string(f"{folder}Data/Latex/{Stock}_df.txt", index=False)

# %%
