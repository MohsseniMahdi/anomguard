from anomguard.ml_logic.registry import load_model
from anomguard.ml_logic.preprocessing import preprocessing_V2_features
import pandas as pd
import pickle
def model_test():
    with open('/home/mahdi/code/MohsseniMahdi/anomguard/models/models/20250318-151719PV2Mlogistic.h5', 'rb') as f:
                    model = pickle.load(f)

    X_pred = pd.read_csv('X_test.csv')
    X_pred_transform = preprocessing_V2_features(X_pred)
    y_pred  = model.predict(X_pred_transform)
    print(y_pred)
    return y_pred



if __name__ == "__main__":
    model_test ()
