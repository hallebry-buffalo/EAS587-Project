import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print("Step 1: Loading dataset...")

df = pd.read_csv("data/processed/WDIcleaned.csv")
print("Dataset loaded successfully.")
print("Shape:", df.shape)

print("\nStep 2: Converting to long format...")
long_df = df.melt(
    id_vars=['Country Name', 'Indicator Name'],
    var_name='Year',
    value_name='Value'
)

long_df['Year'] = long_df['Year'].astype(int)
print("Long format created.")
print(long_df.head().to_string())

print("\nStep 3: Checking indicator names containing 'refugee'...")
refugee_indicators = [i for i in long_df['Indicator Name'].unique() if 'refugee' in i.lower()]

print("Refugee-related indicators found:")
for i in refugee_indicators:
    print(i)

indicator = "Refugees under the mandate of the UNHCR by country or territory of asylum"

print("\nStep 4: Filtering selected indicator...")
indicator_df = long_df[long_df['Indicator Name'] == indicator]
print("Rows found for this indicator:", len(indicator_df))

if indicator_df.empty:
    print("No rows found for this exact indicator name.")
    print("Please copy the exact refugee indicator from the printed list above.")
    exit()

country_name = "India"

print(f"\nStep 5: Filtering country = {country_name} ...")
country_df = indicator_df[indicator_df['Country Name'] == country_name].copy()
print("Rows found for country:", len(country_df))

if country_df.empty:
    print("No rows found for this country and indicator combination.")
    exit()

country_df = country_df.dropna(subset=['Value'])
print("Rows after dropping missing values:", len(country_df))

if country_df.empty:
    print("All values are missing after filtering.")
    exit()

print("\nFiltered data preview:")
print(country_df.head(10).to_string())

print("\nStep 6: Preparing X and y...")
X = country_df[['Year']]
y = country_df['Value']

print("X shape:", X.shape)
print("y shape:", y.shape)

if len(country_df) < 2:
    print("Not enough data points for linear regression.")
    exit()

print("\nStep 7: Training linear regression model...")
model = LinearRegression()
model.fit(X, y)

print("Model trained successfully.")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

print("\nStep 8: Predicting next 2 years...")
last_year = country_df['Year'].max()
future_years = np.array([last_year + 1, last_year + 2]).reshape(-1, 1)
future_predictions = model.predict(future_years)

for year, pred in zip(future_years.flatten(), future_predictions):
    print(f"Predicted refugee value for {year}: {pred:.2f}")

print("\nStep 9: Plotting...")
predicted_values = model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Actual Data")
plt.plot(X, predicted_values, color='red', label="Regression Line")
plt.xlabel("Year")
plt.ylabel("Refugee Value")
plt.title(f"{indicator} in {country_name}")
plt.legend()
plt.tight_layout()
plt.show()

print("Finished successfully.")