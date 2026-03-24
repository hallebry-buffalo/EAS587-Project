import pandas as pd

# Reading raw data 
gu_df = pd.read_csv('data/raw/GROWupRaw.csv')

# Dataset dimensions summary and sample visualization
print(f"Dataset shape: {gu_df.shape[0]} rows × {gu_df.shape[1]} columns")
print(f"Total cells: {gu_df.size}")
print(gu_df.head().to_string())

# Assessing column names and types 
column_info = pd.DataFrame({
    'Column Name': gu_df.columns,
    'Data Type': gu_df.dtypes.values,
    'Non-Null Count': gu_df.count().values,
    'Null Count': gu_df.isnull().sum().values,
    'Null Percentage': (gu_df.isnull().sum() / len(gu_df) * 100).values.round(2)
})
#print(column_info)

# Dropping columns that aren't of interest. We keep columns that indicate the onset of and occurence of armed conflict
keep_cols = ["year","countryname","onset_ko_flag","onset_ko_terr_flag","onset_ko_gov_flag","onset_ko_eth_flag","onset_ko_noneth_flag","onset_ko_terr_eth_flag","onset_ko_gov_eth_flag","onset_ko_terr_noneth_flag","onset_ko_gov_noneth_flag","incidence_flag","incidence_terr_flag","incidence_gov_flag","incidence_eth_flag","incidence_noneth_flag","incidence_terr_eth_flag","incidence_gov_eth_flag","incidence_terr_noneth_flag","incidence_gov_noneth_flag","onset_do_flag","onset_do_terr_flag","onset_do_gov_flag","onset_do_eth_flag","onset_do_noneth_flag","onset_do_terr_eth_flag","onset_do_gov_eth_flag","onset_do_terr_noneth_flag","onset_do_gov_noneth_flag","warhist","warhist_terr","warhist_gov","peaceyears","peaceyears_terr","peaceyears_gov"]

gu_df = gu_df[keep_cols]
# print(gu_df.columns)

# Assessing missing values
total_missing = gu_df.isnull().sum().sum()
total_cells = gu_df.size
missing_percentage = (total_missing / total_cells) * 100

print(f"Total missing values: {total_missing:,}")
print(f"Total cells: {total_cells:,}")
print(f"Missing values percentage: {missing_percentage:.2f}%")

# getting columns where missing values occur
cols_with_missing = gu_df.columns[gu_df.isnull().any()].tolist()
print(f"\nColumns with missing values ({len(cols_with_missing)}):")
print(cols_with_missing)

# Imputing missing values with 0's since all missing values belong to binary categories marking onset of new armed conflict
gu_df.fillna(value=0, inplace=True)

# Re-assessing missing values to check imputation 
total_missing = gu_df.isnull().sum().sum()
total_cells = gu_df.size
missing_percentage = (total_missing / total_cells) * 100

print(f"Total missing values: {total_missing:,}")
print(f"Total cells: {total_cells:,}")
print(f"Missing values percentage: {missing_percentage:.2f}%")

# Merging GROWup data with WDI data 
gu_long = pd.melt(
    gu_df,
    id_vars=['countryname', 'year'],  
    var_name='Indicator Name',  
    value_name='value' 
)

print("Melted dataframe shape:", gu_long.shape)
print("\nFirst few rows:")
print(gu_long.head())
print("\nUnique indicators:", gu_long['Indicator Name'].nunique())

# Pivoting to get years as columns
gu_like_wdi = gu_long.pivot(
    index=['countryname', 'Indicator Name'],
    columns='year',
    values='value'
).reset_index()
gu_like_wdi.columns.name = None 

gu_like_wdi = gu_like_wdi.rename(columns={'countryname': 'Country Name'})

print("\nReshaped dataframe - Structure like World Bank data:")
print("Shape:", gu_like_wdi.shape)
print("\nColumn names (first 10):")
print(gu_like_wdi.columns[:10].tolist())
print("\nYears in data (first 10):")
years = [col for col in gu_like_wdi.columns if isinstance(col, int) or (isinstance(col, str) and col.isdigit())]
print(years[:10])

# Filtering to keep only shared years between GROWup and WDI sets
year_cols = list(range(1960, 2021)) 
keep_cols = ['Country Name', 'Indicator Name'] + [year for year in year_cols if year in gu_like_wdi.columns]
gu_like_wdi_filtered = gu_like_wdi[keep_cols]
gu_like_wdi_filtered.columns = gu_like_wdi_filtered.columns.astype(str)

print(gu_like_wdi_filtered.head(10))

# Imputing NaN values with similar front/backfilling method as WDI
# Imputing missing values 
imputed_df = gu_like_wdi_filtered.copy()

imputed_df.iloc[:, 2:] = imputed_df.iloc[:, 2:].fillna(method='ffill', axis=1)
imputed_df.iloc[:, 2:] = imputed_df.iloc[:, 2:].fillna(method='bfill', axis=1)

# Importing cleaned WDI dataset and merging with GROWup processed data
wdi_df = pd.read_csv('data/processed/WDIcleaned.csv')

combined_df = pd.concat([wdi_df, imputed_df], ignore_index=True)
combined_df = combined_df.sort_values(['Country Name', 'Indicator Name'])
print(combined_df.head(10))

combined_df.to_csv('data/processed/GROWupWDImerged.csv', index=False)