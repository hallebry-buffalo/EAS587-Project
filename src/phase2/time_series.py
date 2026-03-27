import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from views import country_indicator_view

from sklearn.model_selection import train_test_split

import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess   # for TS trend detection
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf   # for residual analysis
from statsmodels.tsa.stattools import adfuller                  # for residual analysis (testing stationarity)
from statsmodels.tsa.arima.model import ARIMA                   # for modeling stochastic component of time series

from scipy import stats
from scipy.stats import probplot

data_path = 'C:/Users/hb102/OneDrive/Documents/GitHub/EAS587-Project/data' # replace as needed 
data = pd.read_csv(data_path + '/processed/GROWupWDImerged.csv')

def get_ts(df, indicator, country):
    '''
    Function to extract single time series for country, indicator row from time series-formatted dataset

    Args:
        df - pandas dataframe
        indicator - string matching target indicator
        country - string matching target country
    
    Returns:
        pandas df formatted as time series (cols = year, indicator)
    '''
    target_row = country_indicator_view(df, country, indicator)

    if target_row.empty:
        raise ValueError(f"No data found for {indicator} in {country}")

    year_cols = [col for col in target_row.columns 
                 if col not in ['indicator', 'country'] 
                 and str(col).isdigit()]
    
    years = pd.to_datetime(year_cols, format='%Y')
    values = target_row[year_cols].iloc[0].values
    
    series = pd.Series(values, index=years, name=f"{indicator} in {country} over time")
    
    # print(series.head(n=5))

    return series


def plot_ts(series, indicator, country, test_size):
    '''
    Function to plot raw time series data

    Args:
        series - pandas series produced by get_ts
        indicator - string, target indicator
        country - string, target country
        test_size - float, proportion of series to use for training data 
    
    Returns:
        no returns, plots time series
    '''
    train, test = train_test_split(series, test_size=test_size, shuffle=False)

    df = train.reset_index()
    df.columns = ['Year', indicator]

    sns.set_theme(style="whitegrid") 

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Year', y=indicator, color='blue')

    plt.xlabel('Year')
    plt.ylabel(indicator)
    plt.title(f'{country} {indicator} Over Time')

    plt.show()


'''
Since we have a dataset that sees changes in trends over time, we need to account for this in our time series forecasts
We will use Locally Weighted Scatterplot Smoothing (LOWESS) as the trend-detection method in this implementation since it is able to capture complex curves
'''
def trend_detect(series, indicator, country, frac=0.5):
    '''
    Function to determine trend component of time series using LOWESS (for annual data)
    
    Args:
        series - pandas series produced by get_ts
        indicator - string, target indicator
        country - string, target country
        frac - float, fraction of data to use for LOWESS smoothing
    
    Returns:
        trend - pandas Series of trend component
        residuals - pandas Series of residual component
    '''
    x = np.arange(len(series))
    y = series.values
    
    # Fiting LOWESS trend curve
    trend_values = lowess(y, x, frac=frac, return_sorted=False)
    trend = pd.Series(trend_values, index=series.index, name='trend')
    
    # Calculating residuals (stochastic component of time series)
    residuals = series - trend
    
    # Plotting time series with trend and residuals
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Original time series with trend
    axes[0].plot(series.index, series, label='Original', color='blue', marker='o', alpha=0.7)
    axes[0].plot(trend.index, trend, label='Trend (LOWESS)', color='red', linewidth=2)
    axes[0].set_title(f'{country}: {indicator} - Training Series with Trend')
    axes[0].set_ylabel(indicator)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Detrended/Residual time series 
    axes[1].plot(residuals.index, residuals, color='green', marker='o')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_title(f'{country}: {indicator} - Residuals (Detrended)')
    axes[1].set_ylabel('Residuals')
    axes[1].grid(True, alpha=0.3)
    
    # Residual histogram (for quick assessment, want to see normality)
    axes[2].hist(residuals.dropna(), bins=15, edgecolor='black', color='green', alpha=0.7)
    axes[2].axvline(x=0, color='red', linestyle='--')
    axes[2].set_title('Residual Distribution')
    axes[2].set_xlabel('Residual Value')
    axes[2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary 
    print(f"\nTrend and Residual Component Summary:")
    print(f"{'='*50}")
    print(f"Trend smoothing fraction: {frac}")
    print(f"Residual mean: {residuals.mean():.4f}")
    print(f"Residual std: {residuals.std():.4f}")
    print(f"Trend explains {100 * (1 - residuals.var() / series.var()):.1f}% of variance")
    
    return trend, residuals

def residual_analysis(residuals, indicator, country):
    '''
    Function for residual analysis to assess properties of detrended series

    Args:
        residuals - pandas Series of residuals from trend_decompose
        indicator - string, target indicator
        country - string, target country
    
    Returns:
        dict - dictionary of diagnostic statistics
    '''
    # Time series plot of residuals
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(residuals.index, residuals.values, color='green', marker='o', linewidth=1)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 0].set_ylabel('Residual Value')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].grid(True, alpha=0.3)

    # Q-Q plot - for assessing normality
    probplot(residuals.values, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Normality Check)')
    axes[0, 1].grid(True, alpha=0.3)
     
    # ACF plot - for determining likley ARIMA model MA component
    plot_acf(residuals.values, ax=axes[1, 0], lags=min(10, len(residuals)//2))
    axes[1, 0].set_title('Autocorrelation Function (ACF)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # PACF plot - for determining likely ARIMA model AR component 
    plot_pacf(residuals.values, ax=axes[1, 1], lags=min(10, len(residuals)//2))
    axes[1, 1].set_title('Partial Autocorrelation Function (PACF)')
    axes[1, 1].grid(True, alpha=0.3)

def res_arima(residuals, max_p=5, max_d=1, max_q=5, ic='aic'):
    '''
    Function to fit ARIMA time series on residuals if residual analysis shows non-white noise process
    
    Args:
        residuals - pandas Series or array of residuals
        max_p - maximum AR parameter to test
        max_d - maximum differencing parameter to test (default is 1 to prevent overdifferencing)
        max_q - maximum MA parameter to test
        ic - information criterion metric for final model selection ('aic' or 'bic' - both are computed)
    
    Returns:
        best_model - fitted ARIMA model
        best_order - tuple of (p,d,q) for best model
        results_df - DataFrame with all model results
    '''
    # Checking if residuals are stationary prior to applying any differencing. Enforces differencing if series non stationary
    adf_result = adfuller(residuals)
    is_stationary = adf_result[1] < 0.05

    if is_stationary:
        dmin = 0
    else:
        dmin = 1
    
    best_score = np.inf
    best_order = None
    best_model = None
    results = []
    
    # Loop through all combinations
    for p in range(max_p + 1):
        for d in range(dmin, max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and d == 0 and q == 0:
                    continue  # Skipping ARIMA(0,0,0) model
                    
                try:
                    model = ARIMA(residuals, order=(p, d, q))
                    fitted = model.fit()
                    
                    # Getting information criterion for model comparison
                    if ic == 'aic':
                        score = fitted.aic
                    elif ic == 'bic':
                        score = fitted.bic
                    else:
                        score = fitted.aic
                    
                    results.append({
                        'parameters': (p, d, q),
                        ic: score,
                        'log_likelihood': fitted.llf
                    })
                    
                    if score < best_score:
                        best_score = score
                        best_order = (p, d, q)
                        best_model = fitted
                        
                except Exception as e:
                    continue
    
    print(f"\nARIMA{best_order} model selected (lowest {ic.upper()}: {best_score:.2f})")
    print(f"\nModel Summary:")
    print(best_model.summary())
    
    # Returning results
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(ic)
    
    return best_model, best_order, results_df

def forecast_with_trend(series, indicator, country, frac=0.5, test_size=0.1, residual_arima = True):
    '''
    Function to forecast using LOWESS trend with train/test validation
    
    Args:
        series - pandas series produced by get_ts
        indicator - string, target indicator
        country - string, target country
        forecast_years - int, number of years to forecast
        frac - float, LOWESS smoothing fraction
        test_size - float, proportion of data to use for testing (0-1)
        residual_arima - boolean, True if residuals should be fitted with ARIMA model 
    
    Returns:
        forecast_series - pandas series with forecasted values
        metrics - dict with validation metrics (if test_size > 0)
    '''
    train, test = train_test_split(series, test_size=test_size, shuffle=False)
    forecast_years = len(test)

    train_trend, train_res = trend_detect(train, indicator, country, frac=0.5)
    
    # Extracting recent trend (previous 5 observations) to use for forecasts
    n_points = 5
    recent_trend = train_trend.values[-n_points:]
    x_recent = np.arange(len(train_trend) - n_points, len(train_trend))
    
    # Fitting linear regression to recent trend
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_recent, recent_trend)
    
    # Plotting validation results
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Calculating residual/non-trend component
    if residual_arima:
        res_model, best_order, results_df = res_arima(train_res, max_p=3, max_d=1, max_q=3, ic='aic')
        forecast = res_model.get_forecast(steps=len(test)) # entire forecast object
        residual_forecast = forecast.predicted_mean # pointwise residual forecasts
        residual_int = forecast.conf_int()
        
        # adding ARIMA residual component to trend component for test period
        x_test = np.arange(len(train), len(train) + len(test))
        trend_test = intercept + slope * x_test
        test_predictions = trend_test + residual_forecast

        lower_bound = trend_test + residual_int.iloc[:, 0]
        upper_bound = trend_test + residual_int.iloc[:, 1]
    
    else:
        # adding median residual component to trend component for test period
        median_residual = np.median(train_res)
        print(f"median_residual: {median_residual}")
        residual_std = np.std(train_res)

        # adding median residual component to trend component for test period
        x_test = np.arange(len(train), len(train) + len(test))
        trend_test = intercept + slope * x_test
        test_predictions = trend_test + median_residual
    
    # Validation metrics
    mae = np.mean(np.abs(test.values - test_predictions))
    rmse = np.sqrt(np.mean((test.values - test_predictions)**2))
    mape = np.mean(np.abs((test.values - test_predictions) / test.values + 1e-8)) * 100     # adding to denominator for numerical stability
        
    # Plotting results 
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Plot 1: Test predictions vs actual observations
    axes[0].plot(test.index, test.values, label='Actual Test Data', color='orange', 
                marker='s', linewidth=2, markersize=6)
    axes[0].plot(test.index, test_predictions, label='Forecast (from training set)', 
                color='red', linestyle='--', marker='x', linewidth=2)

    # Add confidence intervals based on method used
    if residual_arima:
        axes[0].fill_between(test.index, lower_bound, upper_bound,
                            alpha=0.2, color='red', label='95% CI (ARIMA + Trend)')
    else:
        axes[0].fill_between(test.index, 
                            test_predictions - 1.96 * residual_std,
                            test_predictions + 1.96 * residual_std,
                            alpha=0.2, color='red', label='95% CI (from training)')

    axes[0].set_title(f'Test Set Validation: {country} - {indicator}')
    axes[0].set_ylabel(indicator)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Actual vs Predicted (Test set)
    axes[1].scatter(test.values, test_predictions, alpha=0.7, s=50)
    axes[1].plot([test.values.min(), test.values.max()], 
                [test.values.min(), test.values.max()], 
                'r--', alpha=0.5)
    axes[1].set_xlabel('Actual Values')
    axes[1].set_ylabel('Predicted Values')
    axes[1].set_title(f'Actual vs Predicted (Test Set)')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'{country}: {indicator} - Model Validation (Training on {train.index[0].year}-{train.index[-1].year})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    train_test_results = {
        'train': train,
        'test': test,
        'train_trend': pd.Series(train_trend, index=train.index),
        'train_residuals': pd.Series(train_res, index=train.index),
        'test_predictions': pd.Series(test_predictions, index=test.index)
    }
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }

    return metrics, train_test_results


# # Time Series Analysis 1 - Carbon Intensity of GDP in United States 
# indicator1 = 'Carbon intensity of GDP (kg CO2e per constant 2015 US$ of GDP)'
# country1 = 'United States'
# ts = get_ts(data, indicator1, country1)

# plot_ts(ts, indicator1, country1, 0.1)
# train, test = train_test_split(ts, test_size=0.1, shuffle=False)

# trend, residuals = trend_detect(train, indicator1, country1, frac=0.5)

# residual_analysis(residuals, indicator1, country1)

# metrics, results = forecast_with_trend(ts, indicator1, country1, frac=0.5, test_size=0.1, residual_arima=True)

# print(results['test_predictions'].values)
# print(results['test'].values)
# print(metrics)

# # Time Series Analysis 2 - 
# indicator2 = 'Mortality rate, under-5 (per 1,000 live births)'
# country2 = 'West Bank and Gaza'
# ts2 = get_ts(data, indicator2, country2)

# plot_ts(ts2, indicator2, country2, 0.1)
# train2, test2 = train_test_split(ts2, test_size=0.1, shuffle=False)

# trend2, residuals2 = trend_detect(train2, indicator2, country2, frac=0.5)

# residual_analysis(residuals2, indicator2, country2)

# metrics2, results2 = forecast_with_trend(ts2, indicator2, country2, frac=0.5, test_size=0.1, residual_arima=True)

# print(results2['test_predictions'].values)
# print(results2['test'].values)
# print(metrics2)

# # Time Series Analysis 3 - 
# indicator3 = 'Adjusted savings: mineral depletion (% of GNI)'
# country3 = 'Congo, Dem. Rep.'
# ts3 = get_ts(data, indicator3, country3)

# plot_ts(ts3, indicator3, country3, 0.1)
# train3, test3 = train_test_split(ts3, test_size=0.1, shuffle=False)

# trend3, residuals3 = trend_detect(train3, indicator3, country3, frac=0.5)

# residual_analysis(residuals3, indicator3, country3)

# metrics3, results3 = forecast_with_trend(ts3, indicator3, country3, frac=0.5, test_size=0.1, residual_arima=True)

# print(results3['test_predictions'].values)
# print(results3['test'].values)
# print(metrics3)

# Time Series Analysis 4 - 
indicator4 = 'Net migration'
country4 = 'China'
ts4 = get_ts(data, indicator4, country4)

plot_ts(ts4, indicator4, country4, 0.1)
train4, test4 = train_test_split(ts4, test_size=0.1, shuffle=False)

trend4, residuals4 = trend_detect(train4, indicator4, country4, frac=0.5)

residual_analysis(residuals4, indicator4, country4)

metrics4, results4 = forecast_with_trend(ts4, indicator4, country4, frac=0.5, test_size=0.1, residual_arima=True)

print(results4['test_predictions'].values)
print(results4['test'].values)
print(metrics4)