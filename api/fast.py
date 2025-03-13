from anomguard.ml_logic import model, preprocessing, data
from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import io
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

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
        'message': "Hi, The API is running!"
    }

@app.post("/predict")
async def predict_fraud(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Adjust data extraction
        X = df.drop('Class', axis=1)
        y = df['Class']

        # Preprocess the data
        X_train, X_test, y_train_smote = preprocessing.preprocessing_smote(X,X,y)

        # Make predictions
        predictions = model.predict(X_test) #X_test needs to be preprocessed.

        # Convert predictions to a list of dictionaries
        results = [{"prediction": int(p)} for p in predictions]

        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
