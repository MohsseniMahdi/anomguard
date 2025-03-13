## contains model details
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from anomguard.params import *



def initialize_model():
    """
    Initialize the model
    """
    if MODEL_VERSION == 1.0:
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
    else: model = DummyClassifier(strategy= "most_frequent")
    return model



def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val):
    score = model.score(X_val, y_val)
    return score



