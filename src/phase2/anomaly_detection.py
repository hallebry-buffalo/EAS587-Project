# Anomaly Detection Implementation for Phase 2

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from adtk.detector import MinClusterDetector
from adtk.data import validate_series
from adtk.visualization import plot

from sklearn.cluster import KMeans

import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

from views import country_view  # For observing multivariate series for a single country

data_path = 'C:/Users/hb102/OneDrive/Documents/GitHub/EAS587-Project/data' # replace as needed 
data = pd.read_csv(data_path + '/processed/GROWupWDImerged.csv')


def format_data(df, country, ind_subset = None):
    target_country = country_view(df, country)

    if target_country.empty:
        raise ValueError(f"No data found for {country}")

    year_cols = [col for col in target_country.columns 
                if col not in ['Indicator Name', 'Country Name'] 
                and str(col).isdigit()]
    
    melted = target_country.melt(
        id_vars=['Indicator Name', 'Country Name'], 
        value_vars=year_cols,
        var_name='year', 
        value_name='value'
    )

    melted['year'] = pd.to_datetime(melted['year'], format='%Y')
    
    # Transposing df to get indicators as columns and years as indices
    multi_var_df = melted.pivot(
        index='year', 
        columns='Indicator Name', 
        values='value'
    )

    if ind_subset is not None:
        available_indicators = multi_var_df.columns.tolist()
        keep_indicators = [ind for ind in ind_subset if ind in available_indicators]

        multi_var_df = multi_var_df[keep_indicators]

    multi_var_df.columns.name = None 
    multi_var_df.index.name = None  
    
    return multi_var_df

def detrend_multiseries(df, frac):
    '''
    Function to apply LOWESS detrending to each in multivariate series before anomaly detection.
    This accounts for deterministic trends and only looks at residual components to signal anaomalous years
    
    Args:
        df - pandas DataFrame (from format_data output) of multivar time series
        frac - proportion of datapoints to use in LOWESS curve fitting
        
    Returns:
        df - pandas Dataframe of non-deterministic component of multivar time series (better for detecting abnormalities)
    '''
    df_detrended = pd.DataFrame(index=df.index, columns=df.columns)
    
    for col in df.columns:
        series = df[col]

        # Applying LOWESS smoothing as in time_series.py
        x = np.arange(len(series))
        y = series.values
        
        trend_values = lowess(y, x, frac=frac, return_sorted=False)
        trend = pd.Series(trend_values, index=series.index, name='trend')
        
        # Calculating residuals/detrended series
        residuals = series - trend
        
        # Storing detrended sereis
        df_detrended[col] = residuals
    
    return df_detrended

def detect_anom(df, country, ind_subset = None, nclust = 3, frac = 0.5, detrend = True):
    '''
    Function for detecting anomalous years based on a given country's development/conflict indicator set
    
    Args:
        df - pandas DataFrame of multivariate series (original, not detrended)
        country - string, target country for analysis
        ind_subset - list, subset of indicators to consider in anomaly detection
        nclust - int, number of clusters to use in MinClust detection method
        frac - proportion of data to use for LOWESS curve fitting
        detrend - Boolean, True if all time series should be detrended before anomaly detection

    Returns:

    '''
    df = format_data(df, country, ind_subset)

    if detrend:
        df = detrend_multiseries(df, frac=frac)
    
    df = df = validate_series(df)
    min_cluster_detector = MinClusterDetector(KMeans(n_clusters=nclust))
    anomalies = min_cluster_detector.fit_detect(df)

    # Getting z-scores for all data (column/predictor-wise mean and std. dev)
    z_scores = (df - df.mean()) / df.std()

    # Getting average z-score during anomaly years for all predictors
    anomaly_impact = z_scores[anomalies].abs().mean()
    top_5_indicators = anomaly_impact.sort_values(ascending=False).head(5)

    # Plotting top 5 anomalous indicators
    fig, axes = plt.subplots(5, 1, figsize=(12, 10))
    if len(top_5_indicators) == 1:
        axes = [axes]

    for idx, indicator in enumerate(top_5_indicators.index):
        axes[idx].plot(df.index, df[indicator], linewidth=1.5, color='blue')
        
        anomaly_points = df[indicator][anomalies]
        axes[idx].scatter(anomaly_points.index, anomaly_points.values, 
                        color='red', s=50, alpha=0.7)
        
        axes[idx].set_title(f'{indicator} (z-score: {top_5_indicators[indicator]:.2f})')
        axes[idx].set_xlabel('Year')
        axes[idx].set_ylabel('Value')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Defining indicator subsets for more specific anomaly detection - lists generated with help of LLM

trade_commerce = [
    "Agricultural raw materials exports (% of merchandise exports)",
    "Agricultural raw materials imports (% of merchandise imports)",
    "Commercial service exports (current US$)",
    "Commercial service imports (current US$)",
    "Communications, computer, etc. (% of service imports, BoP)",
    "Computer, communications and other services (% of commercial service imports)",
    "Food exports (% of merchandise exports)",
    "Food imports (% of merchandise imports)",
    "Fuel imports (% of merchandise imports)",
    "Manufactures exports (% of merchandise exports)",
    "Manufactures imports (% of merchandise imports)",
    "Merchandise exports (current US$)",
    "Merchandise exports by the reporting economy (current US$)",
    "Merchandise exports by the reporting economy, residual (% of total merchandise exports)",
    "Merchandise exports to economies in the Arab World (% of total merchandise exports)",
    "Merchandise exports to high-income economies (% of total merchandise exports)",
    "Merchandise exports to low- and middle-income economies in East Asia & Pacific (% of total merchandise exports)",
    "Merchandise exports to low- and middle-income economies in Europe & Central Asia (% of total merchandise exports)",
    "Merchandise exports to low- and middle-income economies in Latin America & the Caribbean (% of total merchandise exports)",
    "Merchandise exports to low- and middle-income economies in Middle East & North Africa (% of total merchandise exports)",
    "Merchandise exports to low- and middle-income economies in South Asia (% of total merchandise exports)",
    "Merchandise exports to low- and middle-income economies in Sub-Saharan Africa (% of total merchandise exports)",
    "Merchandise exports to low- and middle-income economies outside region (% of total merchandise exports)",
    "Merchandise exports to low- and middle-income economies within region (% of total merchandise exports)",
    "Merchandise imports (current US$)",
    "Merchandise imports by the reporting economy (current US$)",
    "Merchandise imports by the reporting economy, residual (% of total merchandise imports)",
    "Merchandise imports from economies in the Arab World (% of total merchandise imports)",
    "Merchandise imports from high-income economies (% of total merchandise imports)",
    "Merchandise imports from low- and middle-income economies in East Asia & Pacific (% of total merchandise imports)",
    "Merchandise imports from low- and middle-income economies in Europe & Central Asia (% of total merchandise imports)",
    "Merchandise imports from low- and middle-income economies in Latin America & the Caribbean (% of total merchandise imports)",
    "Merchandise imports from low- and middle-income economies in Middle East & North Africa (% of total merchandise imports)",
    "Merchandise imports from low- and middle-income economies in South Asia (% of total merchandise imports)",
    "Merchandise imports from low- and middle-income economies in Sub-Saharan Africa (% of total merchandise imports)",
    "Merchandise imports from low- and middle-income economies outside region (% of total merchandise imports)",
    "Merchandise imports from low- and middle-income economies within region (% of total merchandise imports)",
    "Ores and metals exports (% of merchandise exports)",
    "Ores and metals imports (% of merchandise imports)",
    "Transport services (% of commercial service imports)",
    "Transport services (% of service imports, BoP)"
]

demographics = [
    "Age dependency ratio (% of working-age population)",
    "Age dependency ratio, old (% of working-age population)",
    "Age dependency ratio, young (% of working-age population)",
    "Population density (people per sq. km of land area)",
    "Population growth (annual %)",
    "Population in largest city",
    "Population in the largest city (% of urban population)",
    "Population in urban agglomerations of more than 1 million (% of total population)",
    "Population, female",
    "Population, female (% of total population)",
    "Population, male",
    "Population, male (% of total population)",
    "Population, total",
    "Rural population",
    "Rural population (% of total population)",
    "Rural population growth (annual %)",
    "Urban population",
    "Urban population (% of total population)",
    "Urban population growth (annual %)",
    "Net migration",
    "Refugees under the mandate of the UNHCR by country or territory of asylum",
    "Refugees under the mandate of the UNHCR by country or territory of origin"
]

climate_emissions = [
    "Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)",
    "Carbon dioxide (CO2) emissions excluding LULUCF per capita (t CO2e/capita)",
    "Carbon dioxide (CO2) emissions from Building (Energy) (Mt CO2e)",
    "Carbon dioxide (CO2) emissions from Fugitive Emissions (Energy) (Mt CO2e)",
    "Carbon dioxide (CO2) emissions from Industrial Combustion (Energy) (Mt CO2e)",
    "Carbon dioxide (CO2) emissions from Industrial Processes (Mt CO2e)",
    "Carbon dioxide (CO2) emissions from Power Industry (Energy) (Mt CO2e)",
    "Carbon dioxide (CO2) emissions from Transport (Energy) (Mt CO2e)",
    "Carbon intensity of GDP (kg CO2e per constant 2015 US$ of GDP)",
    "Methane (CH4) emissions (total) excluding LULUCF (Mt CO2e)",
    "Methane (CH4) emissions from Agriculture (Mt CO2e)",
    "Methane (CH4) emissions from Building (Energy) (Mt CO2e)",
    "Methane (CH4) emissions from Fugitive Emissions (Energy) (Mt CO2e)",
    "Methane (CH4) emissions from Industrial Combustion (Energy) (Mt CO2e)",
    "Methane (CH4) emissions from Power Industry (Energy) (Mt CO2e)",
    "Methane (CH4) emissions from Transport (Energy) (Mt CO2e)",
    "Methane (CH4) emissions from Waste (Mt CO2e)",
    "Nitrous oxide (N2O) emissions (total) excluding LULUCF (Mt CO2e)",
    "Nitrous oxide (N2O) emissions from Agriculture (Mt CO2e)",
    "Nitrous oxide (N2O) emissions from Building (Energy) (Mt CO2e)",
    "Nitrous oxide (N2O) emissions from Fugitive Emissions (Energy) (Mt CO2e)",
    "Nitrous oxide (N2O) emissions from Industrial Combustion (Energy) (Mt CO2e)",
    "Nitrous oxide (N2O) emissions from Industrial Processes (Mt CO2e)",
    "Nitrous oxide (N2O) emissions from Power Industry (Energy) (Mt CO2e)",
    "Nitrous oxide (N2O) emissions from Transport (Energy) (Mt CO2e)",
    "Nitrous oxide (N2O) emissions from Waste (Mt CO2e)",
    "Total greenhouse gas emissions excluding LULUCF (Mt CO2e)",
    "Total greenhouse gas emissions excluding LULUCF per capita (t CO2e/capita)"
]

# Health & Mortality
health = [
    "Adolescent fertility rate (births per 1,000 women ages 15-19)",
    "Birth rate, crude (per 1,000 people)",
    "Death rate, crude (per 1,000 people)",
    "Fertility rate, total (births per woman)",
    "Immunization, DPT (% of children ages 12-23 months)",
    "Immunization, measles (% of children ages 12-23 months)",
    "Life expectancy at birth, female (years)",
    "Life expectancy at birth, male (years)",
    "Life expectancy at birth, total (years)",
    "Mortality rate, adult, female (per 1,000 female adults)",
    "Mortality rate, adult, male (per 1,000 male adults)",
    "Mortality rate, infant (per 1,000 live births)",
    "Mortality rate, infant, female (per 1,000 live births)",
    "Mortality rate, infant, male (per 1,000 live births)",
    "Mortality rate, neonatal (per 1,000 live births)",
    "Mortality rate, under-5 (per 1,000 live births)",
    "Mortality rate, under-5, female (per 1,000 live births)",
    "Mortality rate, under-5, male (per 1,000 live births)",
    "Number of infant deaths",
    "Number of neonatal deaths",
    "Number of under-five deaths",
    "Survival to age 65, female (% of cohort)",
    "Survival to age 65, male (% of cohort)"
]

economic = [
    "GDP (constant 2015 US$)",
    "GDP (constant LCU)",
    "GDP (current LCU)",
    "GDP (current US$)",
    "GDP deflator (base year varies by country)",
    "GDP growth (annual %)",
    "GDP per capita (constant 2015 US$)",
    "GDP per capita (constant LCU)",
    "GDP per capita (current LCU)",
    "GDP per capita (current US$)",
    "GDP per capita growth (annual %)",
    "GNI (current LCU)",
    "GNI (current US$)",
    "GNI per capita (current LCU)",
    "GNI per capita, Atlas method (current US$)",
    "GNI, Atlas method (current US$)",
    "Gross value added at basic prices (GVA) (current US$)",
    "Adjusted net national income per capita (current US$)",
    "Exports of goods and services (% of GDP)",
    "Exports of goods and services (current US$)",
    "External balance on goods and services (% of GDP)",
    "General government final consumption expenditure (% of GDP)",
    "General government final consumption expenditure (current US$)",
    "Gross capital formation (% of GDP)",
    "Gross capital formation (current US$)",
    "Gross domestic savings (% of GDP)",
    "Gross domestic savings (current US$)",
    "Gross fixed capital formation (% of GDP)",
    "Gross fixed capital formation (current US$)",
    "Gross national expenditure (current US$)",
    "Households and NPISHs final consumption expenditure (% of GDP)",
    "Households and NPISHs Final consumption expenditure (current US$)",
    "Imports of goods and services (% of GDP)",
    "Imports of goods and services (current US$)",
    "Final consumption expenditure (current US$)",
    "Trade (% of GDP)",
    "Merchandise trade (% of GDP)",
    "Trade in services (% of GDP)"
]

sectoral_output = [
    "Agriculture, forestry, and fishing, value added (% of GDP)",
    "Agriculture, forestry, and fishing, value added (annual % growth)",
    "Agriculture, forestry, and fishing, value added (constant 2015 US$)",
    "Agriculture, forestry, and fishing, value added (current US$)",
    "Industry (including construction), value added (% of GDP)",
    "Industry (including construction), value added (annual % growth)",
    "Industry (including construction), value added (constant 2015 US$)",
    "Industry (including construction), value added (current US$)",
    "Manufacturing, value added (% of GDP)",
    "Manufacturing, value added (current US$)",
    "Services, value added (% of GDP)",
    "Services, value added (current US$)"
]

natural_resources = [
    "Adjusted savings: energy depletion (% of GNI)",
    "Adjusted savings: energy depletion (current US$)",
    "Adjusted savings: mineral depletion (% of GNI)",
    "Adjusted savings: mineral depletion (current US$)",
    "Adjusted savings: natural resources depletion (% of GNI)",
    "Adjusted savings: net forest depletion (% of GNI)",
    "Coal rents (% of GDP)",
    "Forest rents (% of GDP)",
    "Mineral rents (% of GDP)",
    "Natural gas rents (% of GDP)",
    "Oil rents (% of GDP)",
    "Total natural resources rents (% of GDP)",
    "Agricultural land (% of land area)",
    "Agricultural land (sq. km)",
    "Arable land (% of land area)",
    "Arable land (hectares per person)",
    "Arable land (hectares)",
    "Average precipitation in depth (mm per year)",
    "Land area (sq. km)",
    "Permanent cropland (% of land area)",
    "Renewable internal freshwater resources per capita (cubic meters)",
    "Renewable internal freshwater resources, total (billion cubic meters)",
    "Surface area (sq. km)"
]

# # Anomaly detection 1 - Commerce in South Africa
# detect_anom(data, 'South Africa', trade_commerce, 3)

# # Anomaly detection 2 - Health in Gaza and the West Bank
# detect_anom(data, 'West Bank and Gaza', health, 3)

# Anomaly detection 3 - Emissions in North America
detect_anom(data, 'North America', climate_emissions, 5)
