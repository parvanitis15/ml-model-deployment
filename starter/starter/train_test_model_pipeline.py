"""
This file contains the function for performing the entire machine learning pipeline.
"""
import pickle

from starter.data_segregation import data_segregation
from starter.ml.data import process_data
from starter.ml.model import train_model, inference, compute_model_metrics


def train_test_model_pipeline(data):
    # Data segregation
    train, test = data_segregation(data, test_size=0.20, random_state=8)

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
    # Process the training data with the process_data function
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Save the encoder and label binarizer
    with open('model/encoder.pkl', 'wb') as file:
        pickle.dump(encoder, file)

    with open('model/lb.pkl', 'wb') as file:
        pickle.dump(lb, file)

    # Process the test data with the process_data function
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Train the model
    model = train_model(X_train, y_train)

    # Save the trained model
    with open('model/model.pkl', 'wb') as file:
        pickle.dump(model, file)

    # TEST PERFORMANCE
    print("TRAINING SET")

    # Run inference on the training set
    preds = inference(model, X_train)

    # Compute the accuracy score on the training set
    accuracy = (preds == y_train).mean()
    print(f"Accuracy: {accuracy}")

    # Compute the model metrics
    precision, recall, fbeta = compute_model_metrics(y_train, preds)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F-beta: {fbeta}")

    print("\nVALIDATION SET")

    # Run inference on the validation set
    preds = inference(model, X_test)

    # Compute the accuracy score on the validation set
    accuracy = (preds == y_test).mean()
    print(f"Accuracy: {accuracy}")

    # Compute the model metrics
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F-beta: {fbeta}")
