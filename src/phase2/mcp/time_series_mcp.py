# Phase 2 MCP Implementation - Time Series Forecasting for Country, Indicator Pair


import pandas as pd
import numpy as np

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types
from mcp.server.fastmcp import FastMCP

from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from time_series import get_ts, trend_detect, res_arima, forecast_with_trend 


mcp = FastMCP("forecast-server")

data_path = 'C:/Users/hb102/OneDrive/Documents/GitHub/EAS587-Project/data/processed/GROWupWDImerged.csv'
data = pd.read_csv(data_path)


@mcp.tool()
async def forecast(country, indicator):
    '''
    Function to generate 5-year forecast using forecast function from time_series.py
    '''
    try:
        # Getting target time series from dataframe
        series = get_ts(data, indicator, country)
        
        if series.empty:
            return "No data found for '{indicator}' in '{country}'"
        
        # Fitting deterministic trend and extracting residuals
        trend, residuals = trend_detect(series, indicator, country, frac = 0.5)
        
        
        # Fitting ARIMA on residuals (including testing if ARIMA(0,0,0) is best)
        best_model, best_order, results_df = res_arima(residuals)
        
        # Checking if ARIMA(0,0,0) -- i.e. no ARIMA model -- is the best choice before forecasting
        uses_arima = best_order != (0, 0, 0)
        
        # Forecasting trend component using linear regression on last 5 points
        n_recent = min(5, len(trend))
        recent_trend = trend.values[-n_recent:]
        x_recent = np.arange(len(trend) - n_recent, len(trend))
        
        slope, intercept, _, _, _ = stats.linregress(x_recent, recent_trend)
        
        # Generating 5-year trend forecast
        forecast_years = 5
        x_future = np.arange(len(series), len(series) + forecast_years)
        future_trend = intercept + slope * x_future
        
        # Getting future dates
        future_dates = []
        current_date = series.index[-1]
        for i in range(1, forecast_years + 1):
            future_dates.append(current_date + pd.DateOffset(years=i))
        future_dates = pd.DatetimeIndex(future_dates)
        
        # Generating non-deterministic forecast component based on selected method (ARIMA or simple median residual)
        if uses_arima and best_model:
            # Using ARIMA forecast
            residual_forecast = best_model.forecast(steps=forecast_years)
            future_values = future_trend + residual_forecast.values
            method = f"ARIMA{best_order}"
            method_description = f"ARIMA{best_order} model selected"
        else:
            # Using median residual (no ARIMA)
            future_values = future_trend + np.median(residuals.values)
            method = "Median Residual"
            method_description = "ARIMA(0,0,0) selected (no improvement from residual ARIMA fitting)"
        

        forecast_text = f"5-Year Forecast: {country} - {indicator}\n\n"
        forecast_text += f"Last actual: {series.index[-1].year}: {series.iloc[-1]:.2f}\n\n"
        forecast_text += f"Forecast Method: {method}\n"
        forecast_text += f"Method Description: {method_description}\n\n"
        forecast_text += "Forecast Values:\n"
        
        for date, value in zip(future_dates, future_values):
            forecast_text += f"  • {date.year}: {value:.2f}\n"
        
        # Adding ARIMA metrics if ARIMA was considered
        if uses_arima:
            forecast_text += f"Selected ARIMA Order: {best_order}\n"
            if best_model:
                forecast_text += f"AIC Score: {best_model.aic:.2f}"
        
        return forecast_text
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error generating forecast: {str(e)}"
        )]

@mcp.tool()
async def list_countries():
    '''
    Function to tell users all viable input countries
    '''
    country_col = [col for col in data.columns if 'country' in col.lower()][0]
    countries = sorted(data[country_col].unique().tolist())
    
    text = f"Available Countries ({len(countries)} total):\n\n"
    # Showing top 50
    for country in countries[:50]:
        text += f"• {country}\n"
    if len(countries) > 50:
        text += f"\n... and {len(countries) - 50} more"
    
    return text

@mcp.tool()
async def list_indicators():
    '''
    Function to tell users all viable input indicators
    '''
    indicator_col = [col for col in data.columns if 'indicator' in col.lower()][0]
    indicators = sorted(data[indicator_col].unique().tolist())
    
    text = f"Available Indicators ({len(indicators)} total):\n\n"
    # Top 50
    for indicator in indicators[:50]:
        text += f"• {indicator}\n"
    if len(indicators) > 50:
        text += f"\n... and {len(indicators) - 50} more"
    
    return text


if __name__ == "__main__":
    mcp.run(transport='stdio')