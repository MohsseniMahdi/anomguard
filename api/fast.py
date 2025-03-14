from anomguard.ml_logic import model, preprocessing, data
from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import io
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

from anomguard.ml_logic.registry import load_model
from anomguard.ml_logic.preprocessing import preprocessing_smote
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

#         # 3. Call your prediction function
#         # Assuming 'predict' takes a DataFrame as input:
#         # prediction_result = predict(df)

#         # 4. Remove the temporary file
#         # os.remove(temp_file_path)
#         # model = app.state.model
#         # assert model is not None
#         # X_pred_transform = preprocessing_smote(df)
#         # y_pred = model.predict(X_pred_transform)
#         # return { "prediction": df }

#     except Exception as e:
#         return { "prediction":"prediction_result Error" }
#     #     # 6. Handle any errors that might occur
#     #     if "temp_file_path" in locals():
#     #       os.remove(temp_file_path)
#     #     raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
# # get_predict(file="abc.csv")

    # try:
    #     # Read the uploaded file into Pandas DataFrame
    #     content = await file.read()  # Read file contents as bytes
    #     decoded_content = content.decode("ISO-8859-1")
    #     if file_extension == "csv":
    #         df = pd.read_csv()  # Decode and read as CSV
    #         df = pd.read_csv(io.StringIO(decoded_content), header=None)

        # elif file_extension == "txt":
        #     df = pd.read_table(io.StringIO(content.decode("utf-8")))  # Read as tab-separated file
        # elif file_extension == "json":
        #     df = pd.read_json(io.StringIO(content.decode("utf-8")))  # Read as JSON
        # else:
        #     raise HTTPException(status_code=500, detail=f"Unknown file type {file_extension}")
        # # return {"data_preview": df}
    content = file.file.read()
    print(content)
    df = pd.read_csv(BytesIO(content))
    # print("------\n", df)
    # return json.loads(df.to_json(orient='records'))
    # df.to_json(orient='records')
    # data=BytesIO(content)
    # df = pd.read_csv(data)
    # data.close()
    # file.file.close

    # df = pd.read_csv('test.csv')

    # try:
    model = app.state.model
    assert model is not None

    test = df.drop(columns='Class')
    X_pred_transform = preprocessing_baseline_features(df)
    y_pred = model.predict(X_pred_transform)
    print("******/n", model)
    # return json.loads(df.to_json(orient='records'))
    return { "prediction": y_pred }

    #     # return y_pred
    #     # return {"data_preview": df.head()}  # Return sample of DataFrame
