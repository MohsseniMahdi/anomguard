import glob
import os
import time
import pickle

from typing import Union
from colorama import Fore, Style
from tensorflow import keras

from google.cloud import storage
from anomguard.params import *
from sklearn.base import BaseEstimator


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

def save_model(model:  Union[keras.Model, BaseEstimator] = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally

    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}P{PRE_PROCCESING_VERSION}M{MODEL_VERSION}.h5")

    # model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}P{PRE_PROCCESING_VERSION}M{MODEL_VERSION}.pkl") #this should be changed to h5 when using DL

    pickle.dump(model, open(model_path, 'wb'))

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!

        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None

    return None

def load_model(stage="Production") -> Union[keras.Model, BaseEstimator]:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """
    local_model_directory = os.path.join(LOCAL_REGISTRY_PATH)
    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH)
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)
        try:
            latest_model = keras.models.load_model(most_recent_model_path_on_disk)
        except:
            latest_model = pickle.load(open(most_recent_model_path_on_disk), 'rb')

        print("✅ Model loaded from local disk")

        return latest_model

    elif MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()

        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model")) # changed to models

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            print(f"\n latest model from GCS: {latest_blob.name}")
            # Now proceed to download the model
            latest_model_path_to_save = os.path.join(latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)
            try:
                latest_model = keras.models.load_model(latest_model_path_to_save)
                print("✅ Model loaded using TensorFlow!")
            except:
                with open(latest_model_path_to_save, 'rb') as f:
                    latest_model = pickle.load(f)
                print("✅ Model loaded using Pickle!")

            return latest_model
        except:

            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None
