"""
This is the script that will be used to run the train_model_pipeline function.

Author: P. Arvanitis
"""
import pandas as pd

from starter.train_test_model_pipeline import train_test_model_pipeline

if __name__ == '__main__':
    data_file = 'data/census_clean.csv'

    # Read data
    data = pd.read_csv(data_file)

    # run the pipeline
    train_test_model_pipeline(data)
