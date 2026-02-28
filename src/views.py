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

# Function to filter WDI data by indicator and year to view the distr. of a single indicator in a given year. (e.g. GDP in 1975)
def indicator_by_year(df, indicator, year):
    print(df.columns)
    view = df[df['Indicator Name']==indicator][f'{year}']
    return view 

# Testing functions
wdi_df = pd.read_csv('data/processed/WDIcleaned.csv')

print(indicator_view(wdi_df, 'Commercial service imports (current US$)').head(10))

print(country_view(wdi_df, 'Sudan').head(10))

print(year_view(wdi_df, 1997).head(10))

print(indicator_by_year(wdi_df, 'Adolescent fertility rate (births per 1,000 women ages 15-19)', '1975').head(10))


