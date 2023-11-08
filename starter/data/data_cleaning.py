"""
This script performs data cleaning on census data.

Author: P. Arvanitis
"""

# Open census.csv file, remove all spaces from the column names, and save it as census_clean.csv.
# Hint: Use the pandas library.

import pandas as pd

df = pd.read_csv('census.csv')
df.columns = df.columns.str.replace(' ', '')
df.to_csv('census_clean.csv', index=False)
