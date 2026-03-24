# Algorithm implementation for Phase 2 Workshop

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# from views import country_view
# from views import year_view
# from views import indicator_by_year
from views import indicator_view


data_path = 'C:/Users/hb102/OneDrive/Documents/GitHub/EAS587-Project/data'
data = pd.read_csv(data_path + '/processed/GROWupWDImerged.csv')
#data.head(10)

print(f"Missing values:{data.isnull().sum()}")

indicator_set = data['Indicator Name'].unique()
#print(indicator_set)


year_columns = [col for col in data.columns if col.isdigit()]

def year_stack(df = pd.DataFrame, indicator = str, firstyr = int, lastyr = int):
    '''
    Function to concatenate predictor into a single column for all years
    Args: 
        df - pandas dataframe
        indicator - string, indicator to concat
        firstyr - gives first year in range to concat over
        lastyr - gives last year in range to concat over

    Returns:
        single column view for target indicator in given time period, ready to use for regression and other algs
    '''
    indicator_df = indicator_view(df, indicator)
    
    # Getting the year columns we want
    year_cols = [str(year) for year in range(firstyr, lastyr + 1)]

    melted = pd.melt(
        indicator_df,
        id_vars=['Country Name'],  
        value_vars=year_cols, 
        var_name='Year',
        value_name=indicator 
    )
    
    # Converting year from str to numeric
    melted['Year'] = pd.to_numeric(melted['Year'])
    
    # Dropping missing rows for regression
    melted = melted.dropna(subset=[indicator])

    return melted

def linreg_align(df, features, target, firstyr, lastyr):
    '''
    Function to prepare data for multiple regression
    Returns a DataFrame with all features and target aligned by country and year
    '''
    # List of all indicators we need
    all_indicators = features + [target]

    # Initiating new df on first indicator
    result_df = year_stack(df, all_indicators[0], firstyr, lastyr)
    
    # Merging in all indicators, iteratively filtering to be left with only values which align on country/year
    for indicator in all_indicators[1:]:
        next_df = year_stack(df, indicator, firstyr, lastyr)
        if not next_df.empty:
            result_df = pd.merge(
                result_df,
                next_df,
                on=['Country Name', 'Year'],
                how='inner'
            )
        else:
            print(f"Warning: No data for {indicator}")
    
    return result_df

# Now that we've defined a function to convert time series style data into columns by feature, we can perform multiple linear regression

features = [
        'peaceyears',
        'peaceyears_gov',
        'peaceyears_terr',
        'Life expectancy at birth, total (years)',
        # 'Population growth (annual %)',
        # 'Trade (% of GDP)',
        # 'Inflation, consumer prices (annual %)'
    ]
    
target = 'Refugees under the mandate of the UNHCR by country or territory of origin'

linreg_df = linreg_align(data, features, target, 1960, 2020)

X = linreg_df[features].values
y = linreg_df[target].values

# Splitting data and fitting multiple regression model on training data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg1 = LinearRegression().fit(X_train, y_train)

# Predictions
y_train_pred = reg1.predict(X_train)
y_test_pred = reg1.predict(X_test)

# Performance metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Training R²: {train_r2:.4f}")
print(f"Testing R²: {test_r2:.4f}")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Testing RMSE: {test_rmse:.4f}")
