"""
This file contains a function which computes the performance of a machine learning model on a slice of data
of all categorical features to identify potential biases.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from starter.inference_model_pipeline import inference_model_pipeline
from starter.ml.model import compute_model_metrics


def compute_categorical_features_performance(data, model, cat_features, encoder, lb, save_to_file=True):
    """
    This function computes the performance of a machine learning model on a slice of data
    Parameters
    ----------
    data: dataframe
        Data.
    model: object
        Trained model.
    cat_features: list
        Categorical features.
    encoder: object
        Encoder object.
    lb: object
        Label binarizer object.
    save_to_file: bool
        Indicator to save the results to a file.

    Returns
    -------
    results: dataframe
        Results.
    """
    results = pd.DataFrame(columns=['Feature', 'Value', 'Precision', 'Recall', 'F-beta'])
    for feature in cat_features:
        for value in data[feature].unique():
            subset = data[data[feature] == value]

            # Get X
            X = subset.drop(['salary'], axis=1)

            # Compute model predictions for subset
            preds = inference_model_pipeline(X, model, encoder, lb)

            # Get y
            y = subset['salary']
            # Convert y to binary
            y = lb.transform(y.values).ravel()

            # Compute model metrics for subset
            precision, recall, fbeta = compute_model_metrics(y, preds)

            # Append results
            row = {'Feature': feature, 'Value': value, 'Precision': precision, 'Recall': recall, 'F-beta': fbeta}
            if results.empty:
                results = pd.DataFrame(row, index=[0])
            else:
                results = pd.concat([results, pd.DataFrame(row, index=[0])], ignore_index=True)

        # Plot the f1 results for the feature
        plt.figure(figsize=(10, 5))
        sns.barplot(x='Value', y='F-beta', data=results[results['Feature'] == feature])
        plt.title(f'F-beta for {feature}')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'slice_results/f1/{feature}_fbeta.png')
        plt.close()

        # Plot the precision results for the feature
        plt.figure(figsize=(10, 5))
        sns.barplot(x='Value', y='Precision', data=results[results['Feature'] == feature])
        plt.title(f'Precision for {feature}')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'slice_results/precision/{feature}_precision.png')
        plt.close()

        # Plot the recall results for the feature
        plt.figure(figsize=(10, 5))
        sns.barplot(x='Value', y='Recall', data=results[results['Feature'] == feature])
        plt.title(f'Recall for {feature}')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'slice_results/recall/{feature}_recall.png')
        plt.close()

    if save_to_file:
        results.to_csv('slice_results/slice_output.csv', index=False)

    return results
