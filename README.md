# EAS587-Project
Visvak Sakthivel and Halle Bryant's Project for EAS 587 - Data Intensive Computing 

How to Replicate Phase 1 Results:
 1. Create a virtual environment using command 'python -m venv myenv' on Windows or 'python3 -m venv myenv' on Mac/Linux command line
 2. Activate virtual environment ('myenv\Scripts\activate' on Mac or 'source myenv/bin/activate' on Windows)
 3. Install needed packages by running 'pip install -r requirements.txt' in command line
 4. Run data_cleaning.py script ('python src\data_cleaning.py') to clean WDI raw data
 5. Run growup_cleaning.py ('python src\growup_cleaning.py') to process GROWup sourced data and create merged dataset with WDI data. The combined dataset combines development indicator data and conflict data.
 5. Run exploratory_data_analysis.py ('python src\exploratory_data_analysis.py') to replicate results for EDA.

