"""
This file contains functions to perform data cleaning.

Author: P. Arvanitis
"""


def basic_cleaning(data):
    """
    This function performs basic data cleaning by removing all spaces from the column names.
    Returns
    -------
    data : dataframe
        Cleaned data.
    """
    # remove all spaces from the column names
    data.columns = data.columns.str.replace(' ', '')

    return data
