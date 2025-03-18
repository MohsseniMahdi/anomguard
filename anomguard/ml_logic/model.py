## contains model details
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
        Initialize the logistic regression model
        """
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42)

        return model

def initialize_xgboost():
    """Initialize the XGBoost model"""
    model = XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=1,
        eval_metric="logloss",
        random_state=42,
    )
    return model

def initialize_ensemble(voting_type='soft', max_iter = 1000):
    """Initialize the ensemble model (Voting Classifier).

    Args:
        voting_type (str): 'hard' or 'soft' voting. Defaults to 'soft'.
        max_iter (int): maximum number of iterations.

    Returns:
        VotingClassifier: The initialized VotingClassifier model.
    """
    log_reg_pipe = Pipeline([('scaler', StandardScaler()), ('log_reg', LogisticRegression(random_state=42, max_iter = max_iter))])
    xgb = XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=1,
        eval_metric="logloss",
        random_state=42,
    )

    if voting_type == 'hard':
        model = VotingClassifier(estimators=[('lr', log_reg_pipe), ('xgb', xgb)], voting='hard')
    elif voting_type == 'soft':
        model = VotingClassifier(estimators=[('lr', log_reg_pipe), ('xgb', xgb)], voting='soft')
    else:
        raise ValueError("voting_type must be 'hard' or 'soft'")

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

def evaluate_recall(model, X_test_transformed, y_test):
    y_pred = model.predict(X_test_transformed)
    recall_test = recall_score(y_test, y_pred)
    return {"recall": recall_test}



def evaluate_pr_auc(model, X_test_transformed,y_test):
        y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]
        precision, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc_test = auc(recall_curve, precision)

        return {"pr_auc_test": pr_auc_test}
