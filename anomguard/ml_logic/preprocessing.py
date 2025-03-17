#Import the the libarries
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize


#Package from sklearn
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

#Package from imblearn
from imblearn.over_sampling import SMOTE

def preprocessing_baseline(data):

    '''
    This function performs baseline preprocessing using Robust Scaler on the input data.

    Args:
        data (pd.DataFrame): The training data.

    Returns:
        tuple: A tuple containing the transformed training and testing data.

    '''

    print("Baseline Preprocessing is starting")

    data['Hour'] = (data['Time'] // 3600) % 24

    ## split the data
    X = data.drop(columns = ['Class'])
    y = data['Class']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    rb_scaler = RobustScaler()
    X_train_transformed = rb_scaler.fit_transform(X_train)
    X_test_transformed = rb_scaler.transform(X_test)
    X_val_transformed = rb_scaler.transform(X_val)

    print("Baseline Preprocessing finished")

    return X_train_transformed, X_test_transformed, X_val_transformed, y_train, y_test, y_val


def preprocessing_baseline_features(X):

    '''
    This function performs baseline preprocessing using Robust Scaler on the input data.

    Args:
        data (pd.DataFrame): The training data.

    Returns:
        tuple: A tuple containing the transformed training and testing data.

    '''

    X['Hour'] = (X['Time'] // 3600) % 24

    rb_scaler = RobustScaler()
    X_transformed = rb_scaler.fit_transform(X)


    return X_transformed



def preprocessing_V2(data):

    '''
    This function performs baseline preprocessing using Robust Scaler on the smote data.

    Args:
        data (pd.DataFrame): The  data.



    Returns:
        tuple: A tuple containing the transformed training and testing data.

    '''
    data['Hour'] = (data['Time'] // 3600) % 24

    ## split the data
    X = data.drop(columns = ['Class'])
    y = data['Class']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


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
    X_val_transformed = preprocessor.transform(X_val)

    # Convert back to DataFrame with proper column names

    columns = ['Time', 'Log_Amount', 'Hour_sin', 'Hour_cos'] + [col for col in X_train_smote.columns if col not in ['Time', 'Amount', 'Hour']]
    X_train_transformed = pd.DataFrame(X_train_transformed, columns=columns)

    X_test_transformed = pd.DataFrame(X_test_transformed, columns=columns)

    X_val_transformed = pd.DataFrame(X_val_transformed, columns=columns)


    return X_train_transformed, X_test_transformed, X_val_transformed, y_train_smote, y_test, y_val

def preprocessing_V2_features(X):
    """
    Performs preprocessing_V2 transformations on the input DataFrame.

    Args:
        X (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """

    X['Hour'] = (X['Time'] // 3600) % 24

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
    X_transformed = preprocessor.fit_transform(X)

    # Convert back to DataFrame with proper column names
    columns = ['Time', 'Log_Amount', 'Hour_sin', 'Hour_cos'] + [col for col in X.columns if col not in ['Time', 'Amount', 'Hour']]
    X_transformed = pd.DataFrame(X_transformed, columns=columns)

    return X_transformed


def preprocessing_V3(data):

    '''
    This function performs baseline preprocessing using Robust Scaler on the smote data.

    Args:
        data (pd.DataFrame): The data.



    Returns:
        tuple: A tuple containing the transformed training and testing data.

    '''

    print("preprocessing version 3 is starting")

    df = data.copy()

    duplicate_rows = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)

    df['Hour'] = (df['Time'] // 3600) % 24
    # Apply cyclical transformation
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    df.drop(columns=["Hour"], inplace=True)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print("Original class distribution befor SMOTE in first part:", Counter(y))

    # Apply BorderlineSMOTE (instead of regular SMOTE)
    smote = BorderlineSMOTE(sampling_strategy=0.3, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


    print("Original class distribution after SMOTE in first part:", Counter(y_train_smote))

    #y_train_smote= pd.DataFrame(y_train_smote)

    scaler = RobustScaler()
    X_train_smote.iloc[:, 1:29] = scaler.fit_transform(X_train_smote.iloc[:, 1:29])
    X_test.iloc[:, 1:29] = scaler.transform(X_test.iloc[:, 1:29])
    X_val.iloc[:, 1:29] = scaler.transform(X_val.iloc[:, 1:29])

    columns_to_winsorize = ["V8", "V18", "V21", "V27", "V28"]
    for col in columns_to_winsorize:
        X_train_smote[col] = winsorize(X_train_smote[col], limits=[0.01, 0.01])
        X_test[col] = winsorize(X_test[col], limits=[0.01, 0.01])
        X_val[col] = winsorize(X_val[col], limits=[0.01, 0.01])

    X_train_smote['V20'] = np.log(X_train_smote['V20'].clip(lower=0.0001))
    X_train_smote['V23'] = np.log(X_train_smote['V23'].clip(lower=0.0001))
    X_test['V20'] = np.log(X_test['V20'].clip(lower=0.0001))
    X_test['V23'] = np.log(X_test['V23'].clip(lower=0.0001))
    X_val['V20'] = np.log(X_val['V20'].clip(lower=0.0001))
    X_val['V23'] = np.log(X_val['V23'].clip(lower=0.0001))

    X_train_smote["Amount"] = np.log1p(X_train_smote["Amount"])  # log(1 + Amount) to handle zero values
    X_test["Amount"] = np.log1p(X_test["Amount"])  # log(1 + Amount) to handle zero values
    X_val["Amount"] = np.log1p(X_val["Amount"])  # log(1 + Amount) to handle zero values



    scaler = StandardScaler()
    X_train_smote["Amount"] = scaler.fit_transform(X_train_smote[["Amount"]])
    X_test["Amount"] = scaler.transform(X_test[["Amount"]])
    X_val["Amount"] = scaler.transform(X_val[["Amount"]])

    X_train_smote["Amount"] = winsorize(X_train_smote["Amount"], limits=[0.01, 0.01])
    X_test["Amount"] = winsorize(X_test["Amount"], limits=[0.01, 0.01])
    X_val["Amount"] = winsorize(X_val["Amount"], limits=[0.01, 0.01])

    # Ensure all features are scaled if necessary (PCA is sensitive to feature scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    """n_components = 24
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)"""

    print("Original class distribution befor BorderlineSMOTE:", Counter(y_train_smote))


    # Apply Tomek Links only if class imbalance remains
    tomek = TomekLinks()
    X_final, y_final = tomek.fit_resample(X_train_scaled, y_train_smote)

    # Check final class distribution
    print("After Tomek Links:", Counter(y_final))

    print("preprocessing version 3 finished.")


    return X_final, X_test_scaled, X_val_scaled, y_final, y_test, y_val


def preprocessing_V3_features(X):# -> tuple[Any | DataFrame | ... | Series[Any], Any, Any | Dat...:

    '''
    This function performs baseline preprocessing using Robust Scaler on the smote data.

    Args:
        X (pd.DataFrame): The data.



    Returns:
        tuple: A tuple containing the transformed training and testing data.

    '''

    print("features preprocessing version 3 is starting")


    df = X.copy()

    df['Hour'] = (df['Time'] // 3600) % 24
    # Apply cyclical transformation
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    df.drop(columns=["Hour"], inplace=True)

    #y_train_smote= pd.DataFrame(y_train_smote)

    scaler = RobustScaler()
    df.iloc[:, 1:29] = scaler.fit_transform(df.iloc[:, 1:29])

    columns_to_winsorize = ["V8", "V18", "V21", "V27", "V28"]
    for col in columns_to_winsorize:
        df[col] = winsorize(df[col], limits=[0.01, 0.01])

    df['V20'] = np.log(df['V20'].clip(lower=0.0001))
    df['V23'] = np.log(df['V23'].clip(lower=0.0001))


    df["Amount"] = np.log1p(df["Amount"])  # log(1 + Amount) to handle zero values

    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    df["Amount"] = winsorize(df["Amount"], limits=[0.01, 0.01])

    # Ensure all features are scaled if necessary (PCA is sensitive to feature scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)


    """n_components = 24
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)"""


    print("Preprocessing of features finished")

    return X_scaled
