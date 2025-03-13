# TODO: Import your package, replace this by explicit imports of what you need
from anomguard.interface.main import pred

from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import os
from fastapi.middleware.cors import CORSMiddleware
from anomguard.ml_logic.registry import load_model
from anomguard.ml_logic.preprocessing import preprocessing_smote

app = FastAPI()
app.state.model  = load_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Endpoint for https://your-domain.com/
@app.get("/")
def root():
    return {
        'message': "Hi, The API is running!"
    }

# Endpoint for https://your-domain.com/predict?input_one=154&input_two=199
@app.get("/predict")
def get_predict(X_pred: pd.DataFrame = None):

    model = app.state.model
    assert model is not None
    X_pred_transform = preprocessing_smote(X_pred)
    y_pred = model.predict(X_pred_transform)
    return y_pred
