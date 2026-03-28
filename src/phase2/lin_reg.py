# Linear Regression implementation for Phase 2 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from views import indicator_view


data_path = 'C:/Users/hb102/OneDrive/Documents/GitHub/EAS587-Project/data' # replace as needed 
data = pd.read_csv(data_path + '/processed/GROWupWDImerged.csv')

indicator_set = data['Indicator Name'].unique()
print(indicator_set)

year_columns = [col for col in data.columns if col.isdigit()]

def year_stack(df = pd.DataFrame, indicator = str, firstyr = int, lastyr = int):
    '''
    Function to concatenate predictor into a single column for all country, year observations in a given timeframe
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

def data_align(df, features, target, firstyr, lastyr):
    '''
    Function to prepare data for multiple regression

    Args: 
        df - pandas dataframe
        features - list of features to extract from time series-formatted df
        target - string with name of target feature
        firstyr - gives first year we want to consider in regression
        lastyr - gives last year we want to consider in regression

    Returns:
        Returns a DataFrame with all features and target aligned by country and year
    '''
    # List of all indicators we need to collect
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


def linear_regression(df, features, target, firstyr, lastyr):
    '''
    Function to perform simple/multiple linear regression

    Args: 
        df - pandas dataframe
        features - list of features to extract from time series-formatted df
        target - string with name of target feature
        firstyr - gives first year we want to consider in regression
        lastyr - gives last year we want to consider in regression

    Returns:
        Returns (and prints) test and training RMSE and R^2 metrics
    '''   
    lin_reg_df = data_align(df, features, target, firstyr, lastyr)

    X = lin_reg_df[features].values
    y = lin_reg_df[target].values

    # Splitting data and fitting multiple regression model on training data 
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg_fit = LinearRegression().fit(X_train, y_train)

    # Predictions
    y_train_pred = reg_fit.predict(X_train)
    y_test_pred = reg_fit.predict(X_test)

    # Performance metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred

    print(f"\n\nLinReg Predictors: {features}")
    print(f"LinReg Target: {target}")
    print(f"LinReg Coefficients: {reg_fit.coef_}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Testing R²: {test_r2:.4f}")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Testing RMSE: {test_rmse:.4f}")

    return X_train, X_test, y_train, y_test, train_r2, train_rmse, test_r2, test_rmse, test_residuals, train_residuals


def polynomial_regression(df, features, target, firstyr, lastyr, degree):
    '''
    Function to perform simple/multiple polynomial regression

    Args: 
        df - pandas dataframe
        features - list of features to extract from time series-formatted df
        target - string with name of target feature
        firstyr - gives first year we want to consider in regression
        lastyr - gives last year we want to consider in regression
        degree - integer indicating degree of polynomial curve to be fit

    Returns:
        Returns (and prints) test and training RMSE and R^2 metrics   
    '''   
    poly_reg_df = data_align(df, features, target, firstyr, lastyr)

    X = poly_reg_df[features].values
    y = poly_reg_df[target].values

    # Generating polynomial/interaction features 
    poly = PolynomialFeatures(degree=degree)
    poly_variables = poly.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(poly_variables, y, test_size = 0.2, random_state = 42)

    reg_fit = LinearRegression().fit(X_train, y_train)

    # Predictions
    y_train_pred = reg_fit.predict(X_train)
    y_test_pred = reg_fit.predict(X_test)

    # Performance metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred

    print(f"\n\nPolyReg degree: {degree}")
    print(f"PolyReg Predictors: {features}")
    print(f"PolyReg Target: {target}")
    print(f"PolyReg Coefficients: {reg_fit.coef_}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Testing R²: {test_r2:.4f}")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Testing RMSE: {test_rmse:.4f}")
    return X_train, X_test, y_train, y_test, train_r2, train_rmse, test_r2, test_rmse, test_residuals, train_residuals


def regression_eda(df, features, target, firstyr, lastyr, degree):
    reg_df = data_align(df, features, target, firstyr, lastyr)
    
    # Evaluating interactions and shape of data considering each feature alone
    if degree > 1:
        X_train, X_test, y_train, y_test, train_r2, train_rmse, test_r2, test_rmse, test_residuals, train_residuals = polynomial_regression(df, features, target, firstyr, lastyr, degree)
    
    else:
        X_train, X_test, y_train, y_test, train_r2, train_rmse, test_r2, test_rmse, test_residuals, train_residuals = linear_regression(df, features, target, firstyr, lastyr)

        
    # Residual Histogram (Checking normality of residuals in training data )
    fig, axes = plt.subplots(1, 2, figsize=(12, 10))

    axes[0].hist(train_residuals, bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Training Data Residuals')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Residual Distribution (Assessing normalcy)')

    # Checking for Multicollinearity (VIF > 10 indicates multicollinearity)
    vif_data = pd.DataFrame()
    vif_data['Feature'] = features
    vif_data['VIF'] = [variance_inflation_factor(X_train, i) for i in range(X_train.shape[1])]
    print(vif_data)


# Regression Analysis 1 - Analysing relationship between a country's war history and proportion of population fleeing said country
f1 = [
        'peaceyears_gov',
        'peaceyears_terr',
        'Military expenditure (% of GDP)',
        'warhist',
        'Death rate, crude (per 1,000 people)',
        'GDP per capita (constant 2015 US$)'
    ]
    
t1 = 'Refugees under the mandate of the UNHCR by country or territory of origin'

regression_eda(data, f1, t1, 2015, 2020, 1)

linear_regression(data, f1, t1, 2015, 2020)

# Don't want to consider polynomials above degree 5 since this very likely indicates overfitting
for i in range(2,6):
    polynomial_regression(data, f1, t1, 2015, 2020, i)

# Regression Analysis 2 - Analysing relationship of a country's economic profile and life expectancy
f2 = [
    'Net official development assistance and official aid received (constant 2023 US$)',
    'Foreign direct investment, net inflows (% of GDP)',
    'Final consumption expenditure (current US$)',
    'Official exchange rate (LCU per US$, period average)',
    'GDP per capita (constant 2015 US$)',
    'Net primary income (Net income from abroad) (current US$)'
]

t2 = 'Life expectancy at birth, total (years)'

regression_eda(data, f2, t2, 2010, 2024, 1)

linear_regression(data, f2, t2, 2010, 2024)

for i in range(2,6):
    polynomial_regression(data, f2, t2, 2010, 2024, i)


