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
from anomguard.ml_logic.model import initialize_model, train_model, evaluate_model
from anomguard.ml_logic.registry import save_results, save_model, load_model
from anomguard.ml_logic.data import load_data_to_bq




def preprocess_train():
    query = f"""
        SELECT *
        FROM `{GCP_PROJECT_WAGON}`.{BQ_DATASET}.raw_data

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



    ## split the data
    X = data.drop(columns = ['Class'])
    y = data['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)



    ## performing basic preporccsing
    X_train_transformed, X_val_transformed = preprocessing_baseline(X_train, X_val, y_train)




    model = None
    model = initialize_model()
    model = train_model(model, X_train_transformed, y_train)
    score = evaluate_model(model, X_val_transformed, y_val)

    params = dict()

    save_results(params= params,metrics=dict(score=score))
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
