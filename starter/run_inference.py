"""
This script runs the inference_model_pipeline function.

Author: P. Arvanitis
"""
import pickle

import pandas as pd

from starter.inference_model_pipeline import inference_model_pipeline
from starter.ml.model import compute_model_metrics

if __name__ == '__main__':
    data_file = 'data/census_clean.csv'

    # Read data
    data = pd.read_csv(data_file)

    # Load the model, encoder and label binarizer
    model = pickle.load(open('model/model.pkl', 'rb'))
    encoder = pickle.load(open('model/encoder.pkl', 'rb'))
    lb = pickle.load(open('model/lb.pkl', 'rb'))

    # Get X
    X = data.drop(['salary'], axis=1)

    # run inference
    preds = inference_model_pipeline(X, model, encoder, lb)

    # Get y
    y = data['salary']
    # Convert y to binary
    y = lb.transform(y.values).ravel()

    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y, preds)
    print(f"Accuracy: {(preds == y).mean()}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F-beta: {fbeta}")
