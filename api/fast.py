# TODO: Import your package, replace this by explicit imports of what you need
from anomguard.interface.main_local import predict

from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import os
from fastapi.middleware.cors import CORSMiddleware
# from taxifare.ml_logic.registry import  load_model
# from taxifare.ml_logic.preprocessor import preprocess_features

app = FastAPI()

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
def get_predict():



    return { predict():'greet'}
