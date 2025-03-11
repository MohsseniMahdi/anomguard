import os
import numpy as np


GCP_PROJECT = "<your project id>" # TO COMPLETE
GCP_PROJECT_WAGON = "wagon-public-datasets"
BQ_DATASET = "taxifare"
BQ_REGION = "EU"
MODEL_TARGET = "local"


LOCAL_DATA_PATH =os.environ.get("LOCAL_DATA_PATH")
LOCAL_REGISTRY_PATH = os.environ.get("LOCAL_REGISTRY_PATH")

##################  CONSTANTS  #####################
# LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "MohsseniMahdi", "anomguard","raw_data")
# LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "MohsseniMahdi", "anomguard", "training_outputs")
