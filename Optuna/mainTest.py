# %%
import KNNTest
import LinearTest
import pandas as pd
import RFTest
import SVRTest
from KNNTest import KNNTest
from LinearTest import LinearTest
from RFTest import RFTest
from SVRTest import SVRTest

# import SVRTest
# from SVRTest import SVR_dict

# Run the tests

Stocks = {"^OMXN40", "^OMX", "INVE-B.ST", "VOLV-B.ST", "TELIA.ST", "SOBI.ST", "HM-B.ST"}

Latex_dict = {}

for Stock in Stocks:
    Latex_dict[Stock] = LinearTest(Stock), KNNTest(Stock), SVRTest(Stock), RFTest(Stock)

# %%

Latex_df = pd.DataFrame.from_dict(
    Latex_dict, orient="index", columns=["Linear", "KNN", "SVR", "RF"]
)


# %%
Transposed_df = Latex_df.transpose()

# %%

Transposed_df = Transposed_df.replace(to_replace="\(.*\)", value="", regex=True)

OMXN40_df = Transposed_df["^OMXN40"]
OMX_df = Transposed_df["^OMX"]
INVE_df = Transposed_df["INVE-B.ST"]
VOLV_df = Transposed_df["VOLV-B.ST"]
TELIA_df = Transposed_df["TELIA.ST"]
SOBI_df = Transposed_df["SOBI.ST"]
HM_df = Transposed_df["HM-B.ST"]

# %% export to txt

OMXN40_df.to_string("../Data/Latex/OMXN40_df.txt", index=False)
OMX_df.to_string("../Data/Latex/OMX_df.txt", index=False)
INVE_df.to_string("../Data/Latex/INVE_df.txt", index=False)
VOLV_df.to_string("../Data/Latex/VOLV_df.txt", index=False)
TELIA_df.to_string("../Data/Latex/TELIA_df.txt", index=False)
SOBI_df.to_string("../Data/Latex/SOBI_df.txt", index=False)
HM_df.to_string("../Data/Latex/HM_df.txt", index=False)

# %%
