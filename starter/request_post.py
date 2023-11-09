"""
This script uses the requests module to do one POST request to the API.

Author: P. Arvanitis
"""
import json
import requests
import os
import sys
import pandas as pd
import numpy as np

# Add the root directory of the project to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


if __name__ == '__main__':
    # Load the data
    data = pd.read_csv('data/census_clean.csv')
    # Find a data row with a salary of <=50K
    data = data[data['salary'] == '<=50K'].iloc[0]

    # Get X
    X = data.drop(['salary'])

    # Create a dictionary from the data
    X_dict = dict(X)
    # Convert all numeric values of dict to int
    for key, value in X_dict.items():
        if isinstance(value, np.int64):
            X_dict[key] = int(value)

    # Convert all '-' in keys to '_'
    X_dict = {key.replace('-', '_'): value for key, value in X_dict.items()}

    X_json = json.dumps(X_dict)

    # Make a request to the API
    response = requests.post("https://ml-model-deployment-wh8x.onrender.com/predict", data=X_json)

    # Print the response
    print(response.status_code)
    print(response.json())
