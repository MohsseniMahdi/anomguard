import glob
import os
import time
import pickle

from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage


def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on MLflow
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")

def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")

    return None
