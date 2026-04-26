import pandas as pd

# Reading raw data 
wdi_df = pd.read_csv('data/raw/WDICSV.csv')

# Dataset dimensions summary and sample visualization
print(f"Dataset shape: {wdi_df.shape[0]} rows × {wdi_df.shape[1]} columns")
print(f"Total cells: {wdi_df.size}")
print(wdi_df.head().to_string())

# Assessing column names and types 
column_info = pd.DataFrame({
    'Column Name': wdi_df.columns,
    'Data Type': wdi_df.dtypes.values,
    'Non-Null Count': wdi_df.count().values,
    'Null Count': wdi_df.isnull().sum().values,
    'Null Percentage': (wdi_df.isnull().sum() / len(wdi_df) * 100).values.round(2)
})

# Dropping unnecessary label columns
wdi_df.drop(columns= ['Country Code', 'Indicator Code'], inplace=True)

# Converting years to integers (since we use annual data, this is ideal for easy searching and implementation of views)
new_cols = list(wdi_df.columns)

for i in range(2, len(new_cols)):
    try:
        new_cols[i] = int(float(new_cols[i]))  
    except (ValueError, TypeError):
        pass

wdi_df.columns = new_cols

# Assessing missing values
total_missing = wdi_df.isnull().sum().sum()
total_cells = wdi_df.size
missing_percentage = (total_missing / total_cells) * 100

print(f"Total missing values: {total_missing:,}")
print(f"Total cells: {total_cells:,}")
print(f"Missing values percentage: {missing_percentage:.2f}%")

# Assessing duplicate/constant column issues 
duplicates = wdi_df.duplicated().sum()
print(f"Duplicate rows: {duplicates} ({duplicates/len(wdi_df)*100:.2f}%)")

constant_cols = []
for col in wdi_df.columns:
    if wdi_df[col].nunique() == 1:
        constant_cols.append(col)
if constant_cols:
    print(f"Constant columns (same value in all rows): {constant_cols}")

# Getting top 100 indicators in terms of least proportion of missing values across all countries
row_missing_df = pd.DataFrame({
    'row_index': range(len(wdi_df)),
    'indicator_name': wdi_df['Indicator Name'].values,  # Add indicator name
    'missing_count': wdi_df.isnull().sum(axis=1),
    'missing_percentage': (wdi_df.isnull().sum(axis=1) / wdi_df.shape[1] * 100).round(2)
})

indicators_completeness_df = row_missing_df.groupby('indicator_name').agg({
    'missing_count': ['mean','min', 'max', 'count'],
}).round(2)

indicators_completeness_df.columns = ['_'.join(col).strip() for col in indicators_completeness_df.columns.values] 
indicators_completeness_df = indicators_completeness_df.reset_index()
indicators_completeness_df.sort_values(by='missing_count_mean', ascending=True)

top_500_complete = indicators_completeness_df.nsmallest(500, 'missing_count_mean')

# Removing redundant population indicators (This dataset has over 40 population indicators and we are only interested in a few.)
remove_pop = ['Population ages 0-14 (% of total population)', 'Population ages 0-14, female (% of female population)', 'Population ages 0-14, male (% of male population)', 'Population ages 00-04, female (% of female population)', 'Population ages 00-04, male (% of male population)', 'Population ages 05-09, female (% of female population)', 'Population ages 05-09, male (% of male population)', 'Population ages 10-14, female (% of female population)', 'Population ages 10-14, male (% of male population)', 'Population ages 15-19, female (% of female population)', 'Population ages 15-19, male (% of male population)', 'Population ages 15-64 (% of total population)', 'Population ages 15-64, female (% of female population)', 'Population ages 15-64, male (% of male population)', 'Population ages 20-24, female (% of female population)', 'Population ages 20-24, male (% of male population)', 'Population ages 25-29, female (% of female population)', 'Population ages 25-29, male (% of male population)', 'Population ages 30-34, female (% of female population)', 'Population ages 30-34, male (% of male population)', 'Population ages 35-39, female (% of female population)', 'Population ages 35-39, male (% of male population)', 'Population ages 40-44, female (% of female population)', 'Population ages 40-44, male (% of male population)', 'Population ages 45-49, female (% of female population)', 'Population ages 45-49, male (% of male population)', 'Population ages 50-54, female (% of female population)', 'Population ages 50-54, male (% of male population)', 'Population ages 55-59, female (% of female population)', 'Population ages 55-59, male (% of male population)', 'Population ages 60-64, female (% of female population)', 'Population ages 60-64, male (% of male population)', 'Population ages 65 and above (% of total population)', 'Population ages 65 and above, female (% of female population)', 'Population ages 65 and above, male (% of male population)', 'Population ages 65-69, female (% of female population)', 'Population ages 65-69, male (% of male population)', 'Population ages 70-74, female (% of female population)', 'Population ages 70-74, male (% of male population)', 'Population ages 75-79, female (% of female population)', 'Population ages 75-79, male (% of male population)', 'Population ages 80 and above, female (% of female population)', 'Population ages 80 and above, male (% of male population)', 'Population ages 0-14, female', 'Population ages 0-14, male',
       'Population ages 0-14, total', 'Population ages 15-64, female',
       'Population ages 15-64, male', 'Population ages 15-64, total',
       'Population ages 65 and above, female',
       'Population ages 65 and above, male',
       'Population ages 65 and above, total']

top_500_complete = top_500_complete[~top_500_complete['indicator_name'].isin(remove_pop)]


print("\nTop 250 Most Complete Indicators:")
#print(top_500_complete[['indicator_name', 'missing_count_mean', 'missing_count_min','missing_count_max']].head(500).to_string())
top_250 = top_500_complete['indicator_name'].head(250).to_list()
print(top_250)

# Filtering by top 250 complete indicators and seeing how this changes our missing values metrics
filtered_wdi_df = wdi_df[wdi_df['Indicator Name'].isin(top_250)]

print("\nTop 250 Most Complete Indicators:")
#print(top_500_complete[['indicator_name', 'missing_count_mean', 'missing_count_min','missing_count_max']].head(500).to_string())
print(top_500_complete['indicator_name'].head(250).to_list())

# Dropping cols that still have fewer than 10 years of data available 
filtered_wdi_df = filtered_wdi_df.dropna(thresh = 10)

# Re-assessing missing values
total_missing_filtered = filtered_wdi_df.isnull().sum().sum()
total_cells_filtered = filtered_wdi_df.size
missing_percentage_filtered = (total_missing_filtered / total_cells_filtered) * 100

print(f"Filtered dataset shape: {filtered_wdi_df.shape[0]} rows × {filtered_wdi_df.shape[1]} columns")
print(f"Filtered dataset total missing values: {total_missing_filtered:,}")
print(f"Filtered dataset total cells: {total_cells_filtered:,}")
print(f"Filtered dataset missing values percentage: {missing_percentage_filtered:.2f}%")


# Imputing missing values 
imputed_df = filtered_wdi_df.copy()
imputed_df.iloc[:, 2:] = imputed_df.iloc[:, 2:].ffill(axis=1).bfill(axis=1)

# Checking imputation worked
total_missing_imputed = imputed_df.isnull().sum().sum()
print(f"Imputed dataset total missing values: {total_missing_imputed:,}")

# Converting raw population figures to proportions
metadata_columns = ['Country Name', 'Indicator Name']
year_cols = [col for col in imputed_df.columns if col not in metadata_columns]

indicators = [
    'Primary education, pupils',
    'Secondary education, pupils',
    'Refugees under the mandate of the UNHCR by country or territory of asylum',
    'Refugees under the mandate of the UNHCR by country or territory of origin'
]

for indicator in indicators:
    indicator_rows = imputed_df[imputed_df['Indicator Name'] == indicator]
    
    for idx, row in indicator_rows.iterrows():
        country_name = row['Country Name']
        
        pop_row = imputed_df[(imputed_df['Indicator Name'] == 'Population, total') & 
                             (imputed_df['Country Name'] == country_name)]
        
        if not pop_row.empty:
            for year_col in year_cols:
                if pd.notna(row[year_col]) and pd.notna(pop_row.iloc[0][year_col]) and pop_row.iloc[0][year_col] != 0:
                    imputed_df.loc[idx, year_col] = row[year_col] / pop_row.iloc[0][year_col]

# Saving imputed_df to csv
# imputed_df.to_csv('data/processed/WDIcleaned.csv', index=False)