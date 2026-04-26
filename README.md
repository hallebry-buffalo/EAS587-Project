# EAS587-Project
Visvak Sakthivel and Halle Bryant's Project for EAS 587 - Data Intensive Computing 

How to Replicate Phase 1 Results:
 1. Create a virtual environment using command 'python -m venv myenv' on Windows or 'python3 -m venv myenv' on Mac/Linux command line
 2. Activate virtual environment ('myenv\Scripts\activate' on Mac or 'source myenv/bin/activate' on Windows)
 3. Install needed packages by running 'pip install -r requirements.txt' in command line
 4. Run data_cleaning.py script ('python src\data_cleaning.py') to clean WDI raw data
 5. Run growup_cleaning.py ('python src\growup_cleaning.py') to process GROWup sourced data and create merged dataset with WDI data. The combined dataset combines development indicator data and conflict data.
 5. Run exploratory_data_analysis.py ('python src\exploratory_data_analysis.py') to replicate results for EDA.

 How to Replicate Phase 2 Algorithm Results:
 1. Run all commands outlined above for Phase 1, minus step 5.
 2. Each algorithm belongs to a unique file in /phase2 directory. Run target script using ('python src/script_to_run.py').
 
 Time Series Forecast MCP Server Deployment
 Our MCP server provides 5-year forecasts for World Bank Development Indicators using LOWESS trend detection and ARIMA residual modeling.
 1. Install Prerequisites
    - Python 3.8+
    - Node.js (for MCP Inspector)
 2. Set correct data path in time_series_mcp.py to merged, clean data (i.e.       data_path = 'C:/path/to/your/GROWupWDImerged.csv')
 3. In terminal, run the command 'npx @modelcontextprotocol/inspector python path/to/time_series_mcp.py' to access the MCP inspector in your browser. This will open a web interface at http://localhost:5173 to test each tool.

## Phase 3 Databricks Instructions

Phase 3 notebooks are exported in `src/phase3/`.

1. Upload raw CSV files to the Databricks volume path `/Volumes/workspace/default/raw/`:
   - Raw CSV files must be accessed on UBBox via this link: https://buffalo.box.com/s/qxw3vl7o2ti5y3ap3jjf0527k4bzamnp 
2. Import the notebooks in this order:
   - `src/phase3/bronze_layer.ipynb`
   - `src/phase3/silver_layer.ipynb`
   - `src/phase3/gold_layer.ipynb`
   - `src/phase3/spark_mllib_kmeans.ipynb`
   - `src/phase3/vdem_kmeans.ipynb`
   - `src/phase3/time_series.ipynb`
3. Run `bronze_layer.ipynb` to create the Bronze Delta tables:
   - `bronze_wdi_df`
   - `bronze_growup_df`
   - `bronze_vdem_df`
4. Run `silver_layer.ipynb` to clean, transform, and merge the source tables into:
   - `silver_wdi_df`
   - `silver_growup_df`
   - `silver_vdem_df`
   - `silver_merged_df` (DOES NOT CONTAIN VDEM DATA)
   - `silver_merged_vdem_df` (CONTAINS VDEM DATA)
5. Run `gold_layer.ipynb` to create the Gold summary table:
   - `gold_annual_statistics`
6. Run `spark_mllib_kmeans.ipynb` to train the Spark MLlib KMeans pipeline and save:
   - `gold_kmeans_cv_results`
   - `gold_kmeans_country_clusters`
   - `/Volumes/workspace/default/raw/models/phase3_kmeans_model`
7. Run `vdem_kmeans.ipynb` to train the Spark MLlib KMeans pipeline with V-Dems augmented dataset and save:
   - `/Volumes/workspace/default/raw/models/phase3_kmeans_vdem_model`
8. Run `time_series.ipynb` to train the Spark MLlib time series-esque algorithm using GBT regression and save:
   - `/Volumes/workspace/default/raw/models/phase3_ts_model`
  
The Phase 3 ML notebooks reads from `workspace.default.silver_merged_df` and `workspace.default.silver_merged_vdem_df`, so the Bronze and Silver notebooks must run successfully before the ML notebook.
