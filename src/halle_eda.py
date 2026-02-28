from views import country_view
from views import indicator_view
import pandas as pd

wdi_df = pd.read_csv('data/processed/WDIcleaned.csv')

wdi_df.head()



