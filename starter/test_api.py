"""
This file contains 3 tests for the API: one for the GET and two for the POST method, one that tests each prediction.

Author: P. Arvanitis
"""

import json
import os

import sys
import pandas as pd
import numpy as np

from fastapi.testclient import TestClient

# Import the app from the main.py file
from main import app

# Add the root directory of the project to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a test client for the app
client = TestClient(app)


def test_root():
    """
    This function tests the root endpoint.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the ML Model API!"}


def test_predict_false():
    """
    This function tests the predict endpoint with a false prediction.
    """
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
    response = client.post("/predict", data=X_json)

    print(response.json())

    # Check the response status code
    assert response.status_code == 200

    # Check the response content
    assert response.json() == {"prediction": 0}


def test_predict_successful():
    """
    Test the predict endpoint with a successful prediction.
    """
    # Create a test data dictionary
    test_data = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 77116,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Tech-support",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 2000,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }

    test_data_json = json.dumps(test_data)

    response = client.post("/predict", data=test_data_json)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_invalid_input():
    """
    Test the predict endpoint with invalid input.
    """
    # Create a test data dictionary with invalid input (age is missing)
    test_data = {
        "workclass": "Private",
        "fnlgt": 77116,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Tech-support",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 2000,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }

    test_data_json = json.dumps(test_data)

    response = client.post("/predict", data=test_data_json)
    assert response.status_code == 422
