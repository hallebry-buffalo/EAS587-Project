import pandas as pd

# Functions to filter WDI data by country or by indicator. Ideal view depends on question researchers would like to investigate

def country_view(df, country):
    view = df[df['Country Name']==country]
    return view

def indicator_view(df, indicator):
    view = df[df['Indicator Name']==indicator]
    return view

def year_view(df, year):
    view = df[df['Year'] == year]
    return view 

# Testing functions
wdi_filtered_df = pd.read_csv('data/processed/WDIfiltered.csv')

print(indicator_view(wdi_filtered_df, 'Commercial service imports (current US$)').head(10).to_string())

print(country_view(wdi_filtered_df, 'Sudan').head(10).to_string())


