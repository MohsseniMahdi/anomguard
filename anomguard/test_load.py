from anomguard.ml_logic.registry import load_model
from anomguard.ml_logic.preprocessing import preprocessing_V2_features
import pandas as pd
import pickle
def model_test():
    with open('/home/mahdi/code/MohsseniMahdi/anomguard/models/models/20250318-151719PV2Mlogistic.h5', 'rb') as f:
                    model = pickle.load(f)



    X_pred = X_pred.drop(columns = 'Unnamed: 0')
    
    X_pred_transform = preprocessing_V2_features(X_pred)
   
    y_pred  = model.predict(X_pred)

    return y_pred

def preproccesingv2_check():
    X_pred = pd.read_csv('X_test.csv')
    X_transformed = preprocessing_V2_features(X_pred)
    return X_transformed.shape

if __name__ == "__main__":
    preproccesingv2_check()
