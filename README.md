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
