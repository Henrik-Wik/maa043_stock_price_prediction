# %%
import pandas as pd
from ANNTest import ANNTest
from KNNTest import KNNTest
from LinearTest import LinearTest
from RFTest import RFTest
from SVRTest import SVRTest

# Run the tests

Stocks = {"^OMXSPI", "^OMX", "INVE-B.ST", "VOLV-B.ST", "TELIA.ST", "SOBI.ST", "HM-B.ST"}

# Stocks = {"VOLV-B.ST"}

Latex_dict = {}

for Stock in Stocks:
    # Latex_dict[Stock] = KNNTest(Stock)

    Latex_dict[Stock] = (
        LinearTest(Stock),
        KNNTest(Stock),
        SVRTest(Stock),
        RFTest(Stock),
        ANNTest(Stock),
    )

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
