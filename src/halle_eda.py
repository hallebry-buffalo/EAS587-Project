from views import country_view
from views import indicator_view
from views import year_view
from views import indicator_by_year
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Accessing merged GROWup and WDI data
merged_df = pd.read_csv('data/processed/GROWupWDImerged.csv')

# sns.set_style("whitegrid")
plt.figure(figsize=(10, 8))

# Extracting peaceyears_gov and peaceyears_terr (no. years since military engagement and no. years since experiencing war domestically)
gov_years = merged_df.loc['peaceyears_gov']
terr_years = merged_df.loc['peaceyears_terr']

# Create scatterplot with regression line
sns.scatterplot(data=merged_df, x=gov_years, y=terr_years, alpha=0.6)
sns.regplot(data=merged_df, x=gov_years, y=terr_years, 
            scatter=False, color='red', line_kws={'linestyle': '--'})

plt.xlabel('Peace Years (Government)')
plt.ylabel('Peace Years (Territory)')
plt.title('Comparison of Peace Years: Government vs Territory\nwith Regression Line')
plt.show()



# Exploring distribution of global



