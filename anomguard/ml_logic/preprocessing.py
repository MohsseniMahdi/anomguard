#Import the the libarries
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from collections import Counter
from sklearn.decomposition import PCA
from scipy.stats.mstats import winsorize
import os
from anomguard.params import *



#Package from sklearn
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE


#Package from imblearn
from imblearn.over_sampling import SMOTE


def preprocessing_baseline(data = None, X_test = None):

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
    X_transformed = rb_scaler.transform(X)


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
        X (pd.DataFrame): The input feature DataFrame.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """


    print("*******preprocessing_V2_features is starting********")

    local_data_path = os.path.join(LOCAL_DATA_PATH, 'creditcard.csv')
    df = pd.read_csv(local_data_path)
    df['Hour'] = (df['Time'] // 3600) % 24
    X['Hour'] = (X['Time'] // 3600) % 24

    Xx = df.drop(columns = ['Class'])
    y = df['Class']

    X_train, X_temp, y_train, y_temp = train_test_split(Xx, y, test_size=0.25, random_state=42)
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

    # Define ColumnTransformer to apply transformations
    preprocessor = ColumnTransformer(transformers=[
        ('time_scaler', RobustScaler(), ['Time']),
        ('amount_scaler', RobustScaler(), ['Amount']),  # Scale the log-transformed 'Amount'
        ('hour_cyclical', cyclical_transformer, ['Hour'])
    ], remainder='passthrough')

    X_train_transformed = preprocessor.fit_transform(X_train_smote)
    # Apply the transformation pipeline
    X = X.drop(columns='Unnamed: 0')
    X_transformed = preprocessor.transform(X)

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

    columns_name = X_train_smote.columns

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



    #scaler = StandardScaler()
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

    X_final = pd.DataFrame(X_final, columns=columns_name)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=columns_name)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=columns_name)

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

    df = pd.read_csv('../raw_data/creditcard.csv')
    df = df.drop_duplicates().reset_index(drop=True)

    df['Hour'] = (df['Time'] // 3600) % 24
    # Apply cyclical transformation
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    df.drop(columns=["Hour"], inplace=True)

    Xx = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_temp, y_train, y_temp = train_test_split(Xx, y, test_size=0.25, random_state=42, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print("Original class distribution befor SMOTE in first part:", Counter(y))

    columns_name = X_train.columns

    # Apply BorderlineSMOTE (instead of regular SMOTE)
    smote = BorderlineSMOTE(sampling_strategy=0.3, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    scaler = RobustScaler()
    X_train_smote.iloc[:, 1:29] = scaler.fit_transform(X_train_smote.iloc[:, 1:29])
    X.iloc[:, 1:29] = scaler.transform(X.iloc[:, 1:29])


    columns_to_winsorize = ["V8", "V18", "V21", "V27", "V28"]
    for col in columns_to_winsorize:
        X_train_smote[col] = winsorize(X_train_smote[col], limits=[0.01, 0.01])
        X[col] = winsorize(X[col], limits=[0.01, 0.01])

    X_train_smote['V20'] = np.log(X_train_smote['V20'].clip(lower=0.0001))
    X_train_smote['V23'] = np.log(X_train_smote['V23'].clip(lower=0.0001))

    X['V20'] = np.log(X['V20'].clip(lower=0.0001))
    X['V23'] = np.log(X['V23'].clip(lower=0.0001))


    X_train_smote["Amount"] = np.log1p(X_train_smote["Amount"])  # log(1 + Amount) to handle zero values
    X["Amount"] = np.log1p(X["Amount"])



    #scaler = StandardScaler()
    X_train_smote["Amount"] = scaler.fit_transform(X_train_smote[["Amount"]])
    X["Amount"] = scaler.transform(X[["Amount"]])

    X_train_smote["Amount"] = winsorize(X_train_smote["Amount"], limits=[0.01, 0.01])
    X["Amount"] = winsorize(X["Amount"], limits=[0.01, 0.01])


    # Ensure all features are scaled if necessary (PCA is sensitive to feature scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X = scaler.transform(X)

    """n_components = 24
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)"""

    print("Original class distribution befor BorderlineSMOTE:", Counter(y_train_smote))


    """ # Apply Tomek Links only if class imbalance remains
    tomek = TomekLinks()
    X_final, y_final = tomek.fit_resample(X_train_scaled, y_train_smote)
    """

    """n_components = 24
    pca = PCA(n_components=n_components)
    X = pca.transform(X_scaled)"""



    print("Preprocessing of features finished")

    X = pd.DataFrame(X, columns=columns_name)

    return X

def preprocessing_V4(data):
    """
    Performs preprocessing version 4 on the input data, returning consistent values.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing the transformed training, test, and validation data,
               along with the corresponding target variables.
    """

    print("Preprocessing version 4 is starting")

    df = data.copy()
    df['Hour'] = (df['Time'] // 3600) % 24

    # Separate features and target variable
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Split data into training (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Split training data into training (90%) and validation (10%)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

    # Apply SMOTE to the training set
    smote = SMOTE(sampling_strategy=0.2, random_state=42)  # Adjust ratio if needed
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Initialize RobustScaler
    scaler = RobustScaler()

    # Apply scaling only to 'Time' and 'Amount'
    X_train_smote[['Time', 'Amount']] = scaler.fit_transform(X_train_smote[['Time', 'Amount']])
    X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
    X_val[['Time', 'Amount']] = scaler.transform(X_val[['Time', 'Amount']])

    # Log transform the 'Amount' column to reduce skewness
    X_train_smote['Log_Amount'] = np.log1p(X_train_smote['Amount'])
    X_test['Log_Amount'] = np.log1p(X_test['Amount'])
    X_val['Log_Amount'] = np.log1p(X_val['Amount'])

    # Drop the original 'Amount' column
    X_train_smote.drop(columns=['Amount'], inplace=True)
    X_test.drop(columns=['Amount'], inplace=True)
    X_val.drop(columns=['Amount'], inplace=True)

    # Apply cyclical transformation
    X_train_smote["Hour_sin"] = np.sin(2 * np.pi * X_train_smote["Hour"] / 24)
    X_train_smote["Hour_cos"] = np.cos(2 * np.pi * X_train_smote["Hour"] / 24)
    X_test["Hour_sin"] = np.sin(2 * np.pi * X_test["Hour"] / 24)
    X_test["Hour_cos"] = np.cos(2 * np.pi * X_test["Hour"] / 24)
    X_val["Hour_sin"] = np.sin(2 * np.pi * X_val["Hour"] / 24)
    X_val["Hour_cos"] = np.cos(2 * np.pi * X_val["Hour"] / 24)

    # Drop the 'Hour' column
    X_train_smote.drop(columns=["Hour"], inplace=True)
    X_test.drop(columns=["Hour"], inplace=True)
    X_val.drop(columns=["Hour"], inplace=True)

    X_train_smote['Class'] = y_train_smote

    # Calculate correlation matrix
    correlation_matrix = X_train_smote.corr()

    # Drop low-correlation features
    low_corr_features = ['V26', 'V22', 'V25', 'V23', 'V13', 'Time']
    X_train_smote.drop(columns=low_corr_features, inplace=True)
    X_test.drop(columns=low_corr_features, inplace=True)
    X_val.drop(columns=low_corr_features, inplace=True)

    # Compute absolute correlation with the target column
    target_corr = correlation_matrix['Class'].abs()

    # Find highly correlated pairs
    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [(column, other) for column in upper.columns for other in upper.index if upper[column][other] > 0.85]

    # Drop one feature from each high-correlation pair
    columns_to_drop = []
    for feature1, feature2 in high_corr_pairs:
        if abs(target_corr[feature1]) < abs(target_corr[feature2]):
            columns_to_drop.append(feature1)
        else:
            columns_to_drop.append(feature2)

    X_train_smote.drop(columns=columns_to_drop, inplace=True)
    X_test.drop(columns=columns_to_drop, inplace=True)
    X_val.drop(columns=columns_to_drop, inplace=True)

    X_train_smote.drop(columns=['Class'], inplace=True)

    # 1. Variance Threshold
    vt = VarianceThreshold(threshold=1)
    X_train_vt = vt.fit_transform(X_train_smote)
    selected_vt_features = X_train_smote.columns[vt.get_support()]

    X_test_vt = X_test[selected_vt_features]
    X_val_vt = X_val[selected_vt_features]

    # 2. SelectKBest
    n_features_vt = X_train_vt.shape[1]
    k_value = min(10, n_features_vt)
    k_best = SelectKBest(score_func=f_classif, k=k_value)
    X_train_kb = k_best.fit_transform(X_train_vt, y_train_smote)
    selected_kb_features = selected_vt_features[k_best.get_support()]

    X_test_kb = X_test_vt[selected_kb_features]
    X_val_kb = X_val_vt[selected_kb_features]

    # 3. RFE
    try:
        rfe = RFE(estimator=LogisticRegression(), n_features_to_select=10)
        X_train_rfe = rfe.fit_transform(X_train_kb, y_train_smote)
        selected_rfe_features = selected_kb_features[rfe.support_]
    except Exception as e:
        print(f"Error during RFE: {e}")
        X_train_rfe = X_train_kb
        selected_rfe_features = selected_kb_features

    X_test_rfe = X_test_kb[selected_rfe_features]
    X_val_rfe = X_val_kb[selected_rfe_features]

    # Apply Tomek Links
    tomek = TomekLinks()
    X_train_transformed, y_train = tomek.fit_resample(pd.DataFrame(X_train_rfe, columns=selected_rfe_features), y_train_smote)

    X_test_transformed = pd.DataFrame(X_test_rfe, columns=selected_rfe_features)
    X_val_transformed = pd.DataFrame(X_val_rfe, columns=selected_rfe_features)

    # Check final class distribution
    print("After Tomek Links:", Counter(y_train))

    return X_train_transformed, X_test_transformed, X_val_transformed, y_train, y_test, y_val

def preprocessing_V4_features(X):

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
























































































def preprocessing_V5(df):

    '''
    This function performs baseline preprocessing using Robust Scaler on the input data.

    Args:
        df (pd.DataFrame): The training data.

    Returns:
        tuple: A tuple containing the transformed training and testing data.

    '''

    print("Preprocessing V5 is starting")

    df = df.drop_duplicates().reset_index(drop=True)
    df['Hour'] = (df['Time'] // 3600) % 24

    # Separate features and target variable
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Split data into training and test sets (80-20 split)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Apply SMOTE to the training set
    smote = SMOTE(sampling_strategy=0.25, random_state=42)  # Adjust ratio if needed
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Initialize RobustScaler
    scaler = RobustScaler()
    X_train_smote = scaler.fit_transform(X_train_smote)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    # Convert back to DataFrames
    X_train_smote = pd.DataFrame(X_train_smote, columns=X_train.columns) #Use original column names.
    X_test = pd.DataFrame(X_test, columns=X_train.columns)
    X_val = pd.DataFrame(X_val, columns=X_train.columns)

    # Log transform the 'Amount' column to reduce skewness
    X_train_smote['Log_Amount'] = np.log1p(X_train_smote['Amount'])
    X_test['Log_Amount'] = np.log1p(X_test['Amount'])
    X_val['Log_Amount'] = np.log1p(X_val['Amount'])

    # Drop the original 'Amount' column if needed
    X_train_smote.drop(columns=['Amount'], inplace=True)
    X_test.drop(columns=['Amount'], inplace=True)
    X_val.drop(columns=['Amount'], inplace=True)

    # Apply cyclical transformation
    X_train_smote["Hour_sin"] = np.sin(2 * np.pi * X_train_smote["Hour"] / 24)
    X_train_smote["Hour_cos"] = np.cos(2 * np.pi * X_train_smote["Hour"] / 24)

    X_test["Hour_sin"] = np.sin(2 * np.pi * X_test["Hour"] / 24)
    X_test["Hour_cos"] = np.cos(2 * np.pi * X_test["Hour"] / 24)

    X_val["Hour_sin"] = np.sin(2 * np.pi * X_val["Hour"] / 24)
    X_val["Hour_cos"] = np.cos(2 * np.pi * X_val["Hour"] / 24)

    X_train_smote.drop(columns=["Hour"], inplace=True)
    X_test.drop(columns=["Hour"], inplace=True)
    X_val.drop(columns=["Hour"], inplace=True)

    # Drop low-correlation features
    low_corr_features = ['V26', 'V22', 'V25', 'V23', 'V13', 'Time']
    X_train_smote.drop(columns=low_corr_features, inplace=True)
    X_test.drop(columns=low_corr_features, inplace=True)
    X_val.drop(columns=low_corr_features, inplace=True)

    X_train_smote['Class'] = y_train_smote

    # Compute correlation matrix
    correlation_matrix = X_train_smote.corr()

    # Compute the absolute correlation with the target column
    target_corr = correlation_matrix['Class'].abs()

    # Select upper triangle of correlation matrix to avoid redundancy
    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

    # Find pairs of features with correlation greater than 0.85
    high_corr_pairs = []
    for column in upper.columns:
        high_corr_pairs += [(column, other) for other in upper.index if upper[column][other] > 0.85]

    # For each pair of highly correlated features, drop the one with lower correlation to the target
    columns_to_drop = []
    for feature1, feature2 in high_corr_pairs:
        if abs(target_corr[feature1]) < abs(target_corr[feature2]):
            columns_to_drop.append(feature1)
        else:
            columns_to_drop.append(feature2)

    # Drop the selected columns from X_train_smote
    X_train_smote.drop(columns=columns_to_drop, inplace=True)
    X_test.drop(columns=columns_to_drop, inplace=True)
    X_val.drop(columns=columns_to_drop, inplace=True)

    X_train_smote.drop(columns="Class", inplace=True)

    print("Preprocessing V5 finished")
    print("X_train_smote.shape =", X_train_smote.shape)
    print("X_test.shape =", X_test.shape)
    print("X_val.shape =", X_val.shape)
    print("y_val.shape =", y_val.shape)

    return X_train_smote, X_test, X_val, y_train_smote, y_test, y_val


def preprocessing_V5_features(X):

    '''
    This function performs baseline preprocessing using Robust Scaler on the input data.

    Args:
        X (pd.DataFrame): The training data.

    Returns:
        tuple: A tuple containing the transformed training and testing data.

    '''

    print("Preprocessing V5 featuring is starting")
    df = pd.read_csv('../raw_data/creditcard.csv')
    df = df.drop_duplicates().reset_index(drop=True)
    df['Hour'] = (df['Time'] // 3600) % 24
    X['Hour'] = (X['Time'] // 3600) % 24

    # Separate features and target variable
    Xx = df.drop(columns=['Class'])
    y = df['Class']

    # Split data into training and test sets (80-20 split)
    X_train, X_temp, y_train, y_temp = train_test_split(Xx, y, test_size=0.2, random_state=42, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Apply SMOTE to the training set
    smote = SMOTE(sampling_strategy=0.25, random_state=42)  # Adjust ratio if needed
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Initialize RobustScaler
    scaler = RobustScaler()

    columns_name = X.columns

    # Apply scaling only to 'Time' and 'Amount'
    X_train_smote = scaler.fit_transform(X_train_smote)
    X = scaler.transform(X)

    X = pd.DataFrame(X, columns=columns_name)
    X_train_smote = pd.DataFrame(X_train_smote, columns=X_train.columns)


    # Log transform the 'Amount' column to reduce skewness
    X_train_smote['Log_Amount'] = np.log1p(X_train_smote['Amount'])
    X['Log_Amount'] = np.log1p(X['Amount'])

    # Drop the original 'Amount' column if needed
    X_train_smote.drop(columns=['Amount'], inplace=True)
    X.drop(columns=['Amount'], inplace=True)

    # Apply cyclical transformation
    X_train_smote["Hour_sin"] = np.sin(2 * np.pi * X_train_smote["Hour"] / 24)
    X_train_smote["Hour_cos"] = np.cos(2 * np.pi * X_train_smote["Hour"] / 24)

    X["Hour_sin"] = np.sin(2 * np.pi * X["Hour"] / 24)
    X["Hour_cos"] = np.cos(2 * np.pi * X["Hour"] / 24)

    X_train_smote.drop(columns=["Hour"], inplace=True)
    X.drop(columns=["Hour"], inplace=True)

    # Drop low-correlation features
    low_corr_features = ['V26', 'V22', 'V25', 'V23', 'V13', 'Time']
    X_train_smote.drop(columns=low_corr_features, inplace=True)
    X.drop(columns=low_corr_features, inplace=True)

    X_train_smote['Class'] = y_train_smote

    # Compute correlation matrix
    correlation_matrix = X_train_smote.corr()

    # Compute the absolute correlation with the target column
    target_corr = correlation_matrix['Class'].abs()

    # Select upper triangle of correlation matrix to avoid redundancy
    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

    # Find pairs of features with correlation greater than 0.85
    high_corr_pairs = []
    for column in upper.columns:
        high_corr_pairs += [(column, other) for other in upper.index if upper[column][other] > 0.85]

    # For each pair of highly correlated features, drop the one with lower correlation to the target
    columns_to_drop = []
    for feature1, feature2 in high_corr_pairs:
        if abs(target_corr[feature1]) < abs(target_corr[feature2]):
            columns_to_drop.append(feature1)
        else:
            columns_to_drop.append(feature2)

    # Drop the selected columns from X_train_smote
    X.drop(columns=columns_to_drop, inplace=True)
    X_train_smote.drop(columns=columns_to_drop, inplace=True)

    print("Preprocessing V5 featuring finished")
    print("X.shape =", X.shape)
    #print("X_train_smote.shape =", X_train_smote.shape)



    return X
