from anomguard.ml_logic.registry import load_model
import pandas as pd
def model_test():
    model = load_model()
    X_pred = pd.read_csv('X_test.csv')
    y_pred  = model.predict(X_pred)
    print(y_pred)
    return y_pred



if __name__ == "__main__":
    model_test ()
