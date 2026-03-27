import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Loading processed dataset
df = pd.read_csv('data/processed/WDIcleaned.csv')
print(df.iloc[355158:355162])

# Familiarization with data dimensions and checking for missing values 
df.shape
df.columns
df.info()
df.isna().sum().sum()

df_long = df.melt(
    id_vars=['Country Name','Indicator Name'],
    var_name='Year',
    value_name='Value'
)

# Reconverting years to numeric datatypes from .read_csv() autoconversion to strings
df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')
df_long.head()
df_long.shape
df_long = df_long[(df_long['Year'] >= 1990) & (df_long['Year'] <= 2022)]
df_long['Value'].isna().mean()

# Evaluating indicators by number of missing values (redundant on cleaned/imputed data)
indicator_completeness = (
    df_long.groupby('Indicator Name')['Value']
    .apply(lambda x: x.notna().mean())
    .sort_values(ascending=False)
)
indicator_completeness.head(20)

good_indicators = indicator_completeness[indicator_completeness >= 0.6].index
df_clean = df_long[df_long['Indicator Name'].isin(good_indicators)]
df_clean.shape

# Select indicators to perform preliminary EDA on
target_indicators = [
    "Access to clean fuels and technologies for cooking (% of population)",
    "CO2 emissions (metric tons per capita)",
    "Urban population (% of total population)",
    "Renewable energy consumption (% of total final energy consumption)",
    "GDP per capita (constant 2015 US$)"
]

# Filtering to only sample indicators
df_focus = df_clean[df_clean['Indicator Name'].isin(target_indicators)]

df_pivot = df_focus.pivot_table(
    index=['Country Name','Year'],
    columns='Indicator Name',
    values='Value'
).reset_index()

df_pivot.head()
df_pivot.shape
df_pivot.isna().mean().sort_values(ascending=False)

plt.figure(figsize=(8,5))
sns.heatmap(df_pivot.isna(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

df_pivot.hist(figsize=(12,8), bins=30)
plt.tight_layout()
plt.show()
df_pivot.skew(numeric_only=True)


df_pivot['GDP_log'] = np.log1p(df_pivot['GDP per capita (constant 2015 US$)'])

plt.figure(figsize=(8,6))
sns.heatmap(
    df_pivot.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm',
    fmt=".2f"
)
plt.title("Correlation Matrix")
plt.show()

df_global = df_pivot.groupby('Year').mean(numeric_only=True).reset_index()
df_global.head()

plt.figure(figsize=(10,6))
sns.lineplot(data=df_global, x='Year', y='GDP_log')
plt.title("Global GDP Trend (Log)")
plt.show()

sns.lineplot(data=df_global, x='Year', y='Renewable energy consumption (% of total final energy consumption)')
plt.title("Renewable Energy Consumption")
plt.show()

sns.lineplot(data=df_global, x='Year', y='Access to clean fuels and technologies for cooking (% of population)')
plt.title("Global Clean Fuel Access Trend")
plt.show()

sns.scatterplot(
    data=df_pivot,
    x='GDP_log',
    y='Renewable energy consumption (% of total final energy consumption)'
)
plt.title("GDP vs Renewable energy consumption")
plt.show()

sns.scatterplot(
    data=df_pivot,
    x='GDP_log',
    y='Access to clean fuels and technologies for cooking (% of population)'
)
plt.title("GDP vs Global Clean Fuel Access Trend")
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data=df_pivot[['GDP_log','Access to clean fuels and technologies for cooking (% of population)']])
plt.title("Outlier Detection")
plt.show()


