from starter.ml.data import process_data
from starter.ml.model import inference


def inference_model_pipeline(data, model, encoder, lb):
    """
    This function performs inference on the model.
    Parameters
    ----------
    data: dataframe
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
    X, y, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Run inference
    preds = inference(model, X)

    return preds, y
