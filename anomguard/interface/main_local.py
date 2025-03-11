## file with the model workflow

import numpy as np
import pandas as pd
from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from sklearn.model_selection import train_test_split
from anomgaurd.ml_logic.preprocessing import preprocessing_baseline
from anomgaurd.ml_logic.model import initialize_model, train_model, evaluate_model
from anomgaurd.ml_logic.registry import save_results, save_model




##load_data
data = pd.read_csv('../raw_data/creditcard_data.csv')

## split the data
X = data.drop(columns = ['Class'])
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)



## performing basic preporccsing
X_train_transformed, X_val_transformed = preprocessing_baseline(X_train, X_val)

model = None
model = initialize_model()
model = train_model(model, X_train_transformed, y_train)
score = evaluate_model(model, X_val_transformed, y_val)

save_results(metrics=dict(recall=score))
save_model(model=model)

print("✅ preprocess_and_train() done")
