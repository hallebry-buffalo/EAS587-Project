from views import country_view
from views import indicator_view
import pandas as pd

wdi = pd.read_csv('data/processed/WDIfiltered.csv')

# EDA About Greenland
gl = country_view(wdi, 'Greenland')
print(gl.head(10).to_string())

