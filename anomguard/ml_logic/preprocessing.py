#Import the the libarries
import numpy as np
import pandas as pd

#Package from sklearn
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

#Package from imblearn
from imblearn.over_sampling import SMOTE

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





def preprocessing_smote(X_train, X_test, y_train):

    '''
    This function performs baseline preprocessing using Robust Scaler on the smote data.

    Args:
        X_train (pd.DataFrame): The training data.
        X_test (pd.DataFrame): The testing data.
        y_train (pd.Series): The target variable for the training data.


    Returns:
        tuple: A tuple containing the transformed training and testing data.

    '''

    # Apply SMOTE to the training set
    smote = SMOTE(sampling_strategy=0.2, random_state=42)  # Adjust ratio if needed
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Define function for log transformation
    log_transformer = FunctionTransformer(lambda X: np.log1p(X), validate=False)

    # Define cyclical encoding transformation
    cyclical_transformer = FunctionTransformer(lambda X: np.column_stack((
        np.sin(2 * np.pi * X / 24),
        np.cos(2 * np.pi * X / 24)
    )), validate=False)


    # Define pipeline for 'Amount' - first apply scaling, then log transform
    amount_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('log_transform', log_transformer)
    ])

    # Define ColumnTransformer to apply transformations
    preprocessor = ColumnTransformer(transformers=[
    ('time_scaler', RobustScaler(), ['Time']),  # Scale 'Time' only
    ('amount_pipeline', amount_pipeline, ['Amount']),  # Apply scaling + log transform to 'Amount'
    ('hour_cyclical', cyclical_transformer, ['Hour'])  # Apply sine and cosine encoding to 'Hour'
    ], remainder='passthrough')  # Keep other columns unchanged


    # Apply the transformation pipeline
    X_train_transformed = preprocessor.fit_transform(X_train_smote)
    X_test_transformed = preprocessor.transform(X_test)

    # Convert back to DataFrame with proper column names
    columns = ['Time', 'Log_Amount', 'Hour_sin', 'Hour_cos'] + [col for col in X_train_smote.columns if col not in ['Time', 'Amount', 'Hour']]    X_train_transformed = pd.DataFrame(X_train_transformed, columns=columns)
    X_train_transformed = pd.DataFrame(X_train_transformed, columns=columns)
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=columns)

    return X_train_transformed, X_test_transformed, y_train_smote
