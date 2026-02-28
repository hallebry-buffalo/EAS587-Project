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

