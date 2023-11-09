from starter.ml.data import process_data
from starter.ml.model import inference


def inference_model_pipeline(X, model, encoder, lb):
    """
    This function performs inference on the model.
    Parameters
    ----------
    X: dataframe
        Data.
    model: object
        Trained model.
    encoder: object
        Encoder object.
    lb: object
        Label binarizer object.

    Returns
    -------
    preds: array
        Predictions.
    """
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

    # Process the data with the process_data function
    X, _, _, _ = process_data(
        X, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb
    )

    # Run inference
    preds = inference(model, X)

    return preds
