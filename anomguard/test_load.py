from anomguard.ml_logic.registry import load_model
from anomguard.ml_logic.preprocessing import preprocessing_V2_features
import pandas as pd
def model_test():
    model = load_model()
    X_pred = pd.read_csv('X_test.csv')
    X_pred = X_pred.drop(columns = 'Unnamed: 0')
    y_pred  = model.predict(X_pred)
    print(y_pred)
    return y_pred

def preproccesingv2_check():
    X_pred = pd.read_csv('X_test.csv')
    X_transformed = preprocessing_V2_features(X_pred)
    return X_transformed.shape

if __name__ == "__main__":
    preproccesingv2_check()
