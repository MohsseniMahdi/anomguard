## contains model details
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier



def initialize_model():
    """
    Initialize the model
    """
    model = DummyClassifier(strategy="most_frequent")
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
