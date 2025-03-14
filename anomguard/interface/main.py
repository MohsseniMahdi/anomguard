## file with the model workflow

import numpy as np
import pandas as pd
from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse
from google.cloud import bigquery



from anomguard.params import *

from sklearn.model_selection import train_test_split
from anomguard.ml_logic.preprocessing import preprocessing_baseline
from anomguard.ml_logic.model import *
from anomguard.ml_logic.registry import save_results, save_model, load_model
from anomguard.ml_logic.data import load_data_to_bq




def preprocess_train():
    query = f"""
        SELECT *
        FROM `{GCP_PROJECT}`.{BQ_DATASET}.raw_data

        """

    ##load_data
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("creditcard.csv")
    data_query_cached_exists = data_query_cache_path.is_file()

    if data_query_cached_exists:
            print("Loading data from local CSV...")

            data = pd.read_csv(data_query_cache_path)

    else:
        client = bigquery.Client(project=GCP_PROJECT)
        query_job = client.query(query)
        result = query_job.result()
        data = result.to_dataframe()

        # Save it locally to accelerate the next queries!
        # data.to_csv(data_query_cache_path, header=True, index=False)

    ## performing basic preporccsing
    X_train_transformed, X_test_transformed, y_train, X_val, y_val = preprocessing_baseline(data)

    model = None

    if MODEL_VERSION == "base":
        model = initialize_model()
    elif MODEL_VERSION == "logistic":
        model = initialize_logistic()
    elif MODEL_VERSION == "xgb":
        model = initialize_xgboost()
    else:
        return print("Model version not defined")
    
    print("✅Model loaded")
    
    model = train_model(model, X_train_transformed, y_train)
    print("✅ Model trained")
    
    score = evaluate_model(model, X_val, y_val)
    print("✅ Model evaluated")
    
    pr_auc = evaluate_pr_auc(model, X_test_transformed, y_val)
    print(f"PR AUC score: {pr_auc}")

    params = dict() #TO BE ADDED?

    save_results(params=params, metrics=dict(score=score))
    save_model(model=model)

    print("✅ preprocess_and_train() done")

def load_raw_data():
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("creditcard.csv")
    data = pd.read_csv(data_query_cache_path)

    load_data_to_bq(data, gcp_project=GCP_PROJECT, bq_dataset=BQ_DATASET, table= 'raw_data' , truncate = True)

    print("✅ Loaded successfully to BigQuery")

def pred(X_pred):
   model = load_model()
   model.predict()


if __name__ == '__main__':
    try:

       preprocess_train()
    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
