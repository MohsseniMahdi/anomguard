# TODO: Import your package, replace this by explicit imports of what you need
from packagename.main import predict

from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import os
from fastapi.middleware.cors import CORSMiddleware
from taxifare.ml_logic.registry import  load_model
from taxifare.ml_logic.preprocessor import preprocess_features

app = FastAPI()



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
# @app.post("/predict")
# def get_predict(UploadFile = File(...)):


#     # i.e. feed it to your model.predict, and return the output
#     # For a dummy version, just return the sum of the two inputs and the original inputs

#     allowed_file_types = [
#         "text/csv",
#         "text/plain",
#         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#     ]
#     if file.content_type not in allowed_file_types:
#         # prediction = float(input_one) + float(input_two)
#         # return {
#         #     'prediction': prediction,
#         #     'inputs': {
#         #         'input_one': input_one,
#         #         'input_two': input_two
#         #     }
#         # }

#         try:
#             # Save the file temporarily
#             upload_dir = "uploads"
#             os.makedirs(upload_dir, exist_ok=True)
#             file_path = os.path.join(upload_dir, file.filename)
#             with open(file_path, "wb") as buffer:
#                 shutil.copyfileobj(file.file, buffer)

#                 # Load data into Pandas DataFrame
#             if file.content_type == "text/csv" or file.content_type == "text/plain":
#                 df = pd.read_csv(file_path)
#             elif (
#                 file.content_type
#                 == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             ):
#                 df = pd.read_excel(file_path)
#             else:
#                 raise HTTPException(status_code=500, detail="Error during reading file")

#             # Placeholder for your prediction logic
#             # Replace this with your actual prediction code
#             def predict(df):
#                 # Your logic for prediction using df
#                 # Example:
#                 if "feature1" in df.columns and "feature2" in df.columns:
#                     df['prediction'] = df['feature1'] + df['feature2']
#                     return df['prediction'].tolist()
#                 else:
#                     raise HTTPException(status_code=500, detail="Missing required columns")
#             predictions = predict(df)

#             # Clean up the temporary file (optional)
#             # os.remove(file_path)

#             return {
#                 "predictions": predictions,
#                 "file_name": file.filename,
#                 "message": "Prediction successful",
#             }

#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
