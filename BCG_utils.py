from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import numpy as np
import pickle

# Dummy test code parts
# df = pd.read_csv("customer_data.csv")
# df = df.iloc[:1000]
# X = df.drop("churned", axis=1).values
# y = df["churned"].values

# clf = DummyClassifier()
# clf.fit(X, y)

# with open('clf.pickle', 'wb') as f:
#     pickle.dump(clf, f)

def model_statistics(model_pickle, X, y):
    skf = StratifiedKFold(n_splits=5)
    
    with open(model_pickle, 'rb') as f:
        clf = pickle.load(f)
    # print(cross_val_score(clf, X, y, scoring="f1", cv=5))

    for train_index, test_index in skf.split(X, y):
        _, X_test = X[train_index], X[test_index]
        _, y_test = y[train_index], y[test_index]
        print(classification_report(y_test, clf.predict(X_test)))
        
# model_statistics('clf.pickle', X, y)

def return_predictions(model_pickle, X):
    """Return the model probability predictions for class 1 as an array
    """
    with open(model_pickle, 'rb') as f:
        clf = pickle.load(f)
        
    y_hat = clf.predict_proba(X)
    # X["y_hat"] = clf.predict_proba(X)
    return y_hat[:,1]
    
# return_predictions('clf.pickle', X)    