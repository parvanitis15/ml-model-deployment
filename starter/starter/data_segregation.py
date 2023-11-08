"""
This file contains functions for data segregation: train and test data split.
"""
from sklearn.model_selection import train_test_split


def data_segregation(data, test_size=0.20):
    """
    This function performs train and test data split.
    Returns
    -------
    train : dataframe
        Training data.
    test : dataframe
        Test data.
    """
    train, test = train_test_split(data, test_size=test_size)

    return train, test
