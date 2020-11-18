from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

import pickle
import pandas as pd
import numpy as np

df = pd.read_csv("customer_data.csv")

features_list = ["transaction_count", "transaction_value"]
X = df[features_list]
y = np.array(df["churned"])

def model_statistics(model_pickle, X, y):
    skf = StratifiedKFold(n_splits=5)
    
    with open(model_pickle, 'rb') as f:
        clf = pickle.load(f)

    for train_index, test_index in skf.split(X, y):
        _, X_test = X[train_index], X[test_index]
        _, y_test = y[train_index], y[test_index]
        print(classification_report(y_test, clf.predict(X_test)))
        
def return_predictions(model_pickle, X):
    """Return the model probability predictions for class 1 as an array
    """
    with open(model_pickle, 'rb') as f:
        clf = pickle.load(f)
        
    y_hat = clf.predict_proba(X)
    
    return y_hat[:,1]

model_statistics(r"F:\GitHub\BCG_hack_2020\bcg_hack\first_weighted_logistic.sav", X.values, y)
df["y_hat"] = return_predictions(r"F:\GitHub\BCG_hack_2020\bcg_hack\first_weighted_logistic.sav", X)