## contains model details
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_recall_curve, auc
from sklearn.dummy import DummyClassifier
from anomguard.params import *


def initialize_model():
    """
    Initialize the model
    """
    model = DummyClassifier(strategy="most_frequent")

    return model

def initialize_logistic():
        """
        Initialize the model
        """
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42)

        return model

def initialize_xgboost():
    """XGBoost."""
    model = XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=1,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# def compile_model(model: Model, learning_rate=0.0005) -> Model:
#     """
#     Compile the Neural Network
#     """
#     pass

def evaluate_model(model, X_val, y_val):
    score = model.score(X_val, y_val)
    return score

def evaluate_recall(y_test, y_pred):
    recall_test = recall_score(y_test, y_pred)
    return {"recall": recall_test}



def evaluate_pr_auc(model, X_test_transformed,y_test):
        y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]
        precision, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc_test = auc(recall_curve, precision)

        return {"pr_auc_test": pr_auc_test}
