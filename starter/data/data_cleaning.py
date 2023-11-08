"""
This script performs data cleaning on census data.

Author: P. Arvanitis
"""

# Open census.csv file, remove all spaces and save it as census_clean.csv

with open('census.csv', 'r') as file:
    lines = file.readlines()

    # Remove spaces
    lines = [line.replace(' ', '') for line in lines]

    # Write to census_clean.csv
    with open('census_clean.csv', 'w') as file:
        for line in lines:
            file.write(line)
