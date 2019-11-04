# https://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/analyses/churn.ipynb

from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF

churn_df = pd.read_csv('./data/churn.csv', sep=',')
churn_df.head()

col_names = churn_df.columns.tolist()
col_names
to_show = col_names[:6] + col_names[-6:]
to_show
churn_df[to_show].head(6)

# Isolate target data
churn_result = churn_df['Churn?']
y = np.where(churn_result == 'True.', 1, 0)
to_drop = ['State', 'Area Code', 'Phone', 'Churn?']
churn_feat_space = churn_df.drop(to_drop, axis=1)
churn_feat_space.head(5)

# yes / no has to be converted to boolean values
yes_no_cols = ['Int\'l Plan', 'VMail Plan']
churn_feat_space.loc[:, yes_no_cols] = churn_feat_space.loc[:, yes_no_cols] == 'yes'

# pull out features for future use
features = churn_feat_space.columns
features

X = churn_feat_space.as_matrix().astype(np.float)

# This is important?
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(f'Feature space holds {X.shape[0]} and {X.shape[1]} features')
print(f'Unique target labels: ', np.unique(y))

def run_cv(X, y, clf_class, **kwargs):
    kf = KFold(shuffle=True)
    kf.get_n_splits(X)
    y_pred = y.copy()

    # Iterate through folds
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # initialize a classifier with keyword arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

from sklearn.linear_model import LogisticRegression as LR

print('Logistic Regression \n', accuracy(y, run_cv(X, y, LR)))
