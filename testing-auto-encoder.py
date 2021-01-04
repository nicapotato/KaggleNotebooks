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
from sklearn import preprocessing

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
df = pd.concat([X,test],axis=0)
del X, test; gc.collect()

# df.is_null()
# missing = df.isnull().sum().reset_index()
# df = df.dropna(thresh=df.shape[0]*0.75, axis=1)
df.fillna(-999, inplace=True)
# from sklearn.feature_selection import SelectKBest, f_classif,chi2
# selector = SelectKBest(f_classif, k=150)
# selector.fit(X, y)
# X.isnull().sum().sum()

# test = SelectKBest(chi2, k=2).fit_transform(X, y)

# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.datasets import load_iris
# from sklearn.feature_selection import SelectFromModel
# print(X.shape)

# clf = ExtraTreesClassifier()
# clf = clf.fit(X, y)
# clf.feature_importances_  

# model = SelectFromModel(clf, prefit=True)
# X_new = model.transform(X)
# X_new.shape  


# Final Train and Test Set
cat_cols = ['Main_NAME_CONTRACT_TYPE', 'Main_CODE_GENDER', 'Main_FLAG_OWN_CAR', 'Main_FLAG_OWN_REALTY', 'Main_NAME_TYPE_SUITE',
    'Main_NAME_INCOME_TYPE', 'Main_NAME_EDUCATION_TYPE', 'Main_NAME_FAMILY_STATUS', 'Main_NAME_HOUSING_TYPE', 'Main_OCCUPATION_TYPE',
    'Main_WEEKDAY_APPR_PROCESS_START', 'Main_ORGANIZATION_TYPE', 'Main_FONDKAPREMONT_MODE', 'Main_HOUSETYPE_MODE', 'Main_WALLSMATERIAL_MODE',
    'Main_EMERGENCYSTATE_MODE']
X = df.loc[traindex,:]
print("Train Set Shape: {} Rows, {} Columns".format(*X.shape))
feat_names = X.columns
test = df.loc[testdex,:]
print("Submission Set Shape: {} Rows, {} Columns".format(*test.shape))
del df; gc.collect();

print("Modeling")
modelstart = time.time()
VALID = True
n_rounds = 8000
xgb_params = {'eta': 0.01, 
              'max_depth': 6, 
              'subsample': 0.8, 
              'colsample_bytree': 0.1,
              'min_child_weight' : 35,
              #'scale_pos_weight': ,
              'objective': 'binary:logistic', 
              'eval_metric': 'auc', 
              'seed': 23,
              'lambda': 2,
              'alpha': 1,
              'silent': 1
             }


if VALID is True:
    # Training and Validation Set
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y,test_size=0.15, random_state=23)
    del X, y; gc.collect();
    
    # XGBOOST Efficient Feature Storage
    d_train = xgb.DMatrix(X_train, y_train,feature_names=feat_names)
    d_valid = xgb.DMatrix(X_valid, y_valid,feature_names=feat_names)
    d_test = xgb.DMatrix(test,feature_names=feat_names)
    
    modelstart = time.time()
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    model = xgb.train(xgb_params, d_train, n_rounds, watchlist, verbose_eval=150, early_stopping_rounds=200)
    xgb_pred = model.predict(d_test)
    
    del d_train, d_valid, d_test; gc.collect();
else:
    # XGBOOST Efficient Feature Storage
    d_train = xgb.DMatrix(X, y,feature_names=feat_names)
    d_test = xgb.DMatrix(test,feature_names=feat_names)
    
    modelstart = time.time()
    watchlist = [(d_train, 'train')]
    model = xgb.train(xgb_params, d_train, n_rounds, watchlist, verbose_eval=150, early_stopping_rounds=200)
    xgb_pred = model.predict(d_test)

# Submit
xgb_sub = pd.DataFrame(xgb_pred,columns=["TARGET"],index=testdex)
xgb_sub.index.rename("SK_ID_CURR",inplace=True)
xgb_sub.to_csv("XGB.csv",index=True,float_format='%.8f')
print("\Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
print("\nNotebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))