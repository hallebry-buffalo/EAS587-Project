import pandas as pd

# Functions to filter WDI data by country or by indicator. Ideal view depends on question researchers would like to investigate

def country_view(df, country):
    view = df[df['Country Name']==country]
    return view

def indicator_view(df, indicator):
    view = df[df['Indicator Name']==indicator]
    return view

def year_view(df, year):
    view = df[['Country Name', 'Indicator Name', f'{year}']]
    return view 

# Testing functions
wdi_df = pd.read_csv('data/processed/WDIcleaned.csv')

print(indicator_view(wdi_df, 'Commercial service imports (current US$)').head(10).to_string())

print(country_view(wdi_df, 'Sudan').head(10).to_string())

print(year_view(wdi_df, 1997).head(10).to_string())


