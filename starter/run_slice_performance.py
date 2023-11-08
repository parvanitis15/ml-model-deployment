"""
This is the script that will be used to run the slice_performance function.
"""

import pickle
import pandas as pd

from starter.slice_performance import compute_categorical_features_performance

if __name__ == '__main__':
    # Load the data
    data = pd.read_csv('data/census_clean.csv')

    # Load the model, encoder and label binarizer
    model = pickle.load(open('model/model.pkl', 'rb'))
    encoder = pickle.load(open('model/encoder.pkl', 'rb'))
    lb = pickle.load(open('model/lb.pkl', 'rb'))

    # Get the categorical features
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Compute the performance of the model on a slice of the data
    results = compute_categorical_features_performance(data, model, cat_features, encoder, lb)
