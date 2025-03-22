from anomguard.ml_logic import model, preprocessing, data
from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import io
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

from anomguard.params import *

from anomguard.ml_logic.registry import load_model
from anomguard.ml_logic.preprocessing import preprocessing_baseline_features, preprocessing_V2_features, preprocessing_V3_features
from anomguard.ml_logic.preprocessing import preprocessing_V4_features, preprocessing_V5_features
from io import BytesIO

from typing import Annotated


app = FastAPI()
app.state.model  = load_model()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    return {
        'message': "Hi, running!"
    }


# Endpoint for https://your-domain.com/predict?input_one=154&input_two=199
@app.post("/predict")
async def get_predict(file: UploadFile = File(...)):
#     """
#     Endpoint to receive a file, process it with the predict function, and return the result.
#     """

    content = file.file.read()
    df = pd.read_csv(BytesIO(content))


    model = app.state.model

    assert model is not None
    X = df.drop(columns='Unnamed: 0')


    print("PRE_PROCCESING_of_Features_VERSION", PRE_PROCCESING_VERSION)
        ## performing basic preporccsing
    if PRE_PROCCESING_VERSION == "V1":
        X_pred_transform = preprocessing_baseline_features(X)
    elif PRE_PROCCESING_VERSION == "V2":
        X_pred_transform = preprocessing_V2_features(X)
    elif PRE_PROCCESING_VERSION == "V3":
        X_pred_transform = preprocessing_V3_features(X)
    elif PRE_PROCCESING_VERSION == "V4":
        X_pred_transform = preprocessing_V4_features(X)
    elif PRE_PROCCESING_VERSION == "V5":
        X_pred_transform = preprocessing_V5_features(X)
    else:
        print("Wrong version of preprocessing for prediction is selected")


    y_pred = model.predict(X_pred_transform)
    return { "prediction": y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred }
