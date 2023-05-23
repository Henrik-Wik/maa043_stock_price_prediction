# %%
import pandas as pd
from ANNTest import ANNTest
from KNNTest import KNNTest
from LinearTest import LinearTest
from RFTest import RFTest
from SVRTest import SVRTest

# Run the tests

folder = "../"

Stocks = {"^OMXSPI", "^OMX", "INVE-B.ST", "VOLV-B.ST", "TELIA.ST", "SOBI.ST", "HM-B.ST"}

Latex_dict = {}

for Stock in Stocks:

    Latex_dict[Stock] = (
        LinearTest(Stock, folder),
        KNNTest(Stock, folder),
        SVRTest(Stock, folder),
        RFTest(Stock, folder),
        ANNTest(Stock, folder),
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
