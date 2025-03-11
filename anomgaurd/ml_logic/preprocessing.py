#Import the the libarries
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split

def preprocessing_baseline(X_train, X_test):

    '''
    This function performs baseline preprocessing using Robust Scaler on the input data.

    Args:
        X_train (pd.DataFrame): The training data.
        X_test (pd.DataFrame): The testing data.

    Returns:
        tuple: A tuple containing the transformed training and testing data.

    '''
    

    rb_scaler = RobustScaler()
    X_train_transformed = rb_scaler.fit_transform(X_train)
    X_test_transformed = rb_scaler.transform(X_test)

    return X_train_transformed, X_test_transformed
