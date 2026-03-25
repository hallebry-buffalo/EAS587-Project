import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.model_selection import train_test_split

def fit_ts(ts, test_size):
    # Splitting series into training and test sets, maintaining order
    train, test = train_test_split(ts, test_size = 0.1, shuffle = False)
    model = ARIMA(ts, order=(1, 1, 1))
    model_fit = model.fit(disp = 0)
    forecast = model_fit.forecast(steps = 7)[0]