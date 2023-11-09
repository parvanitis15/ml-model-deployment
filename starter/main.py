"""
This file contains the FastAPI app and the code for the API, with a GET method to welcome the user
and a POST method to perform inference.

Author: P. Arvanitis
"""
# Put the code for your API here.
from fastapi import FastAPI
import pandas as pd
import pickle
from pydantic import BaseModel

from starter.inference_model_pipeline import inference_model_pipeline

app = FastAPI()


class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 25,
                    "workclass": "Private",
                    "fnlgt": 77116,
                    "education": "Bachelors",
                    "education_num": 13,
                    "marital_status": "Never-married",
                    "occupation": "Tech-support",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital_gain": 2000,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "native_country": "United-States"
                }
            ]
        }
    }


@app.get("/")
async def root():
    return {"message": "Welcome to the ML Model API!"}


@app.post("/predict")
async def predict(data: CensusData):
    """
    This function performs inference on the model.
    Parameters
    ----------
    data: dataframe
        Data.

    Returns
    -------
    preds: array
        Predictions.
    """
    # Load the model, encoder and label binarizer
    with open('model/model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('model/encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)
    with open('model/lb.pkl', 'rb') as file:
        lb = pickle.load(file)

    # Convert data to dict
    data = data.dict()

    # Convert all '_' in keys to '-'
    data = {key.replace('_', '-'): value for key, value in data.items()}

    # Create a dataframe from the data
    X = pd.DataFrame([data])

    # Run inference
    preds = inference_model_pipeline(X, model, encoder, lb)

    # return {"data": data}
    return {"prediction": int(preds[0])}
