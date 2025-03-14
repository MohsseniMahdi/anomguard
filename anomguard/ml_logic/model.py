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
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
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
    
if MODEL_VERSION == 0.0:

elif MODEL_VERSION == 1.0:
    
elif MODEL_VERSION == 2.0:

 

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# def compile_model(model: Model, learning_rate=0.0005) -> Model:
#     """
#     Compile the Neural Network
#     """
#     pass

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val):
    score = model.score(X_val, y_val)
    return score


        # recall_test = recall_score(y_test, y_pred_test)

        # # PR-AUC (Precision-Recall AUC) на train
        # y_probs_train = model.predict_proba(X_train_transformed)[:, 1]
        # precision, recall_curve, _ = precision_recall_curve(y_train_smote, y_probs_train)
        # pr_auc_train = auc(recall_curve, precision)
        
        # # PR-AUC на test
        # y_probs_test = model.predict_proba(X_test_transformed)[:, 1]
        # precision, recall_curve, _ = precision_recall_curve(y_test, y_probs_test)
        # pr_auc_test = auc(recall_curve, precision)
        
        # return {
        #     "recall_train": recall_train, "recall_test": recall_test,
        #     "pr_auc_train": pr_auc_train, "pr_auc_test": pr_auc_test
        # }    
