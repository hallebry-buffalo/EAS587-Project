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