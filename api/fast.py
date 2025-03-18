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
#from sklearn.dummy import DummyClassifier
from io import BytesIO
#import raw_data
#import shutil

import json

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
    print('content', content)
    df = pd.read_csv(BytesIO(content))
    print('======', df.head())

    # try:
    model = app.state.model

    assert model is not None
    print("******/n", df.columns)
    X = df.drop(columns='Unnamed: 0')
    print("******/n", X.columns)
    # test = df.drop(columns='Class')

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
    # return json.loads(df.to_json(orient='records'))
    # return { "prediction": y_pred }
    print("******!!!!!******/n", y_pred)

    return { "prediction": y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred }

    #     # return {"data_preview": df.head()}  # Return sample of DataFrame



 # print("------\n", df)
    # return json.loads(df.to_json(orient='records'))
    # df.to_json(orient='records')
    # data=BytesIO(content)
    # df = pd.read_csv(data)
    # data.close()
    # file.file.close

    # df = pd.read_csv('test.csv')





   # if not file:
    #     raise HTTPException(status_code=400, detail="No file provided")

    # # # Check file type (optional, you can add more robust checks here)
    # allowed_filetypes = ["csv", "txt", "json"]
    # file_extension = file.filename.split(".")[-1].lower()
    # # return {
    # #     'message': "Hi, The postman is running!"
    # # }

    # if file_extension not in allowed_filetypes:
    #     raise HTTPException(status_code=400, detail="Invalid file type. Allowed types are: csv, txt, json")
        # return { "prediction":"adsad" }
#     try:

#         #1. Save the uploaded file to a temporary location
#         # with raw_data(delete=False, suffix=f".{file_extension}") as tmp_file:
#         #     shutil.copyfileobj(file.file, tmp_file)
#         #     temp_file_path = tmp_file.name
#         # return { "prediction":"prediction_result try" }

#         # 2. Load the data from the file into a pandas DataFrame (or other format)

#         if file_extension == "csv":
#             # df = pd.read_csv(file.file)
#             # return { "prediction": pd.read_csv(file)}
#             df = pd.read_csv(io.StringIO(content.decode("utf-8")))
#             return { "prediction": file.file }

#         # elif file_extension == "txt":
#         #     df = pd.read_table(file.file)
#         # elif file_extension == "json":
#         #     df = pd.read_json(file.file)
#         # else:
#         #     raise HTTPException(status_code=500, detail=f"unknown file type {file_extension}")
