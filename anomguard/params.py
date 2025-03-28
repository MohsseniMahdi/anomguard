import os

GCP_PROJECT = os.environ.get("GCP_PPROJECT") # TO COMPLETE
# GCP_PROJECT_WAGON = "wagon-public-datasets"
# GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON")

MODEL_TARGET = os.environ.get("MODEL_TARGET")
PRE_PROCCESING_VERSION =  os.environ.get("PRE_PROCCESING_VERSION")
MODEL_VERSION = os.environ.get("MODEL_VERSION")

GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")

INSTANCE = os.environ.get("INSTANCE")

# MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
# MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
# MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
# PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
# PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")
# EVALUATION_START_DATE = os.environ.get("EVALUATION_START_DATE")
# GAR_IMAGE = os.environ.get("GAR_IMAGE")
# GAR_MEMORY = os.environ.get("GAR_MEMORY")


LOCAL_DATA_PATH =os.environ.get("LOCAL_DATA_PATH")
LOCAL_REGISTRY_PATH = os.environ.get("LOCAL_REGISTRY_PATH")

##################  CONSTANTS  #####################
# LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "MohsseniMahdi", "anomguard","raw_data")
# LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "MohsseniMahdi", "anomguard", "training_outputs")
