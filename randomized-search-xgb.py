# Classification CatBoost for Home Credit Default Risk Kaggle Competition
# By Nick Brooks, May 2018 [Other Kaggler's work also used, citation throughout]
# https://www.kaggle.com/c/home-credit-default-risk

import time
notebookstart = time.time()

# General
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

# Viz
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing
import scipy.stats as st
import warnings

print("READ CSV")
X = pd.read_csv("../input/grand-features/X.csv", index_col="Main_SK_ID_CURR")#[:1000]
feat_names = X.columns
y = pd.read_csv("../input/grand-features/y.csv",header=None)#[:1000]
y.columns = ["SK_ID_CURR","TARGET"]
y = y["TARGET"].values
test = pd.read_csv("../input/grand-features/test.csv",index_col="Main_SK_ID_CURR")#.sample(1000)
traindex = X.index
testdex = test.index

print("MODELING")
# Training and Validation Set
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,test_size=0.10, random_state=23)
del X, y; gc.collect();

params = { 
      "n_estimators": 500,
      "learning_rate": 0.3,
      #'subsample': 0.8, 
      #'colsample_bytree': 0.75,
      #'min_child_weight' : 25,
      'seed': 23,
      }

param_grid = { 
      "max_depth": st.randint(3, 10),
      'scale_pos_weight': st.randint(1, 13),
      'reg_alpha': st.randint(1, 5)
      }

fit_dict = {"eval_set":[(X_train, y_train),(X_valid, y_valid)],
          "early_stopping_rounds":20,
          "eval_metric":"auc",
          "verbose":100}

alg = xgb.XGBClassifier(**params)
print("Model Parameters: ", alg.get_params().keys())
clf = RandomizedSearchCV(estimator = alg,n_iter = 4, param_distributions=param_grid, cv=2, scoring = "roc_auc")

print("Parameter Search:")
clf.fit(X_train,y_train, **fit_dict)

print("Best All Params: ",clf.get_params())
print("Best Score: ",clf.best_score_)
print("Best Parametes: ", clf.best_params_)
xgb_pred = clf.predict_proba(test)[:,1]
xgb_pred[:5]

# Submit
xgb_sub = pd.DataFrame(xgb_pred,columns=["TARGET"],index=testdex)
xgb_sub.to_csv("XGB.csv",index=True,float_format='%.8f')
print("\nNotebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))