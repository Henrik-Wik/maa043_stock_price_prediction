
# %%
import pandas as pd
import finnhub

from yahooquery import Ticker

df = pd.DataFrame(Ticker('MSFT').summary_detail)

finnhub_client = finnhub.Client(api_key='cg10unpr01qpqqs2e130cg10unpr01qpqqs2e13g')


#%%

df = finnhub_client.company_basic_financials('AMZN', metric='all')
