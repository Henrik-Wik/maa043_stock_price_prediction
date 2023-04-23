# %%
import pandas as pd
from KNNTest import KNNTest
from LinearTest import LinearTest
from RFTest import RFTest
from SVRTest import SVRTest
from ANNTest import ANNTest

# Run the tests

Stocks = {"^OMXSPI", "^OMX", "INVE-B.ST", "VOLV-B.ST", "TELIA.ST", "SOBI.ST", "HM-B.ST"}

# Stocks = {"INVE-B.ST", "VOLV-B.ST", "TELIA.ST", "SOBI.ST", "HM-B.ST"}

Latex_dict = {}

for Stock in Stocks:
    Latex_dict[Stock] = ANNTest(Stock)

    # Latex_dict[Stock] = LinearTest(Stock), KNNTest(Stock), SVRTest(Stock), RFTest(Stock), ANNTest(Stock)

# %%

# Latex_df = pd.DataFrame.from_dict(
#     Latex_dict, orient="index", columns=["Linear", "KNN", "SVR", "RF"]
# )

Latex_df = pd.DataFrame.from_dict(
    Latex_dict, orient="index", columns=["ANN"]
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
    dfs[Stock].to_string(f"../Data/Latex/{Stock}_df.txt", index=False)

# %%
