import time
start = time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
# models
from sklearn.linear_model import LogisticRegression
from sklearn import feature_selection
import xgboost as xgb
from xgboost.sklearn import XGBClassifier # <3

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import preprocessing
from scipy.sparse import hstack, csr_matrix
import scipy.stats as st

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py

import gc
gc.enable()
import os

train = pd.read_csv("../input/donorschoose-application-screening/train.csv",index_col="id",low_memory=False,
                    parse_dates=["project_submitted_datetime"])#.sample(120000,random_state=23)
train = pd.merge(train,pd.read_csv("../input/dense-text-feature-engineering/nicks_train.csv",index_col="id"),left_index=True,right_index=True,how = "left")
train.shape

traindex = train.index
test = pd.read_csv("../input/donorschoose-application-screening/test.csv",index_col="id",low_memory=False,
                    parse_dates=["project_submitted_datetime"])#.sample(1000)
test = pd.merge(test,pd.read_csv("../input/dense-text-feature-engineering/nicks_test.csv",index_col="id"),left_index=True,right_index=True,how = "left")
test.shape
tesdex = test.index
project_is_approved = train["project_is_approved"].copy()
df = pd.concat([train.drop("project_is_approved",axis=1),test],axis=0)
alldex = df.index

del test, train

print("Creating Features:")
# Feature Engineering
df['text'] = df.apply(lambda row: ' '.join([
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']), 
    str(row['project_essay_4'])]), axis=1)
df = pd.merge(df, df["project_subject_categories"].str.get_dummies(sep=', '),
              left_index=True, right_index=True, how="left")
df = pd.merge(df, df["project_subject_subcategories"].str.get_dummies(sep=', '),
              left_index=True, right_index=True, how="left")
              
text_cols = ["project_essay_1","project_essay_2","project_essay_3","project_essay_4","project_resource_summary",
             "project_title","description","text"]

print("\nTime Vars:")
df["Year"] = df["project_submitted_datetime"].dt.year
df["Date of Year"] = df['project_submitted_datetime'].dt.dayofyear # Day of Year
df["Weekday"] = df['project_submitted_datetime'].dt.weekday
df["Day of Month"] = df['project_submitted_datetime'].dt.day
df["Quarter"] = df['project_submitted_datetime'].dt.quarter

# Dummies or encoder..
timevars = ['Weekday','Day of Month','Year','Date of Year',"Quarter"]
encode = ["teacher_id"] # 'teacher_prefix','school_state','project_grade_category'
lbl = preprocessing.LabelEncoder()
for col in encode:
     df[col] = lbl.fit_transform(df[col].astype(str))

df = pd.get_dummies(df, columns=['teacher_prefix','school_state','project_grade_category']+timevars)
#df.drop(timevars,axis=1,inplace=True)

df.drop(['project_subject_categories',"project_subject_subcategories",
         "project_submitted_datetime"],axis=1,inplace=True)
normalize = ["teacher_number_of_previously_posted_projects","quantity","price"]
gc.collect()
objcols = df.loc[:,df.dtypes == object].columns

print("Creating Word Features Matrix")
# build TFIDF Vectorizer
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    stop_words= "english",
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1,1),
    dtype=np.float32,
    max_features=5000,
    norm='l2',
    min_df=0,
    smooth_idf=False)
    
word_vectorizer.fit(df.loc[traindex,"text"])
training = word_vectorizer.transform(df.loc[traindex,'text'])
testing = word_vectorizer.transform(df.loc[tesdex,'text'])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    stop_words= "english",
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1,1),
    dtype=np.float32,
    max_features=550,
    norm='l2',
    min_df=0,
    smooth_idf=False)
    
for cols in ["project_resource_summary","project_title"]:#,"description"]:
    word_vectorizer.fit(df.loc[traindex,cols])
    temp = word_vectorizer.transform(df.loc[:,cols])
    training = hstack([training,word_vectorizer.transform(df.loc[traindex,cols])])
    testing = hstack([testing,word_vectorizer.transform(df.loc[tesdex,cols])])
#df["description"].head()
df.drop(objcols,axis=1,inplace=True)
gc.collect()
df.head()

X = hstack([training,csr_matrix(df.loc[traindex,:].values)])
y = project_is_approved.copy()
testing = hstack([testing,csr_matrix(df.loc[tesdex,:].values)])
for shape in [X,testing]:
    print("{} Rows and {} Cols".format(*shape.shape))
del df, project_is_approved, training
gc.collect()

print("Modeling..")
cv_scores = []
xgb_preds = []

#for train_index, test_index in kf.split(X_train_stack):
seed = 23
# Split out a validation set
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=seed)

xgb_params = {'eta': 0.2, 
              'max_depth': 8, 
              'subsample': 0.8, 
              'colsample_bytree': 0.8,
              'min_child_weight' : 1.5,
              'scale_pos_weight':y.value_counts()[0]/y.value_counts()[1],
              'objective': 'binary:logistic', 
              'eval_metric': 'auc', 
              'seed': seed,
              'lambda': 1.5,
              'alpha': .5
             }

d_train = xgb.DMatrix(X_train, y_train)
d_valid = xgb.DMatrix(X_valid, y_valid)
d_test = xgb.DMatrix(testing)

watchlist = [(d_valid, 'valid')]
model = xgb.train(xgb_params, d_train, 700, watchlist, verbose_eval=50, early_stopping_rounds=30)
cv_scores.append(float(model.attributes()['best_score']))
xgb_pred = model.predict(d_test)
xgb_preds.append(list(xgb_pred))
sub = pd.DataFrame(xgb_pred,columns=["project_is_approved"],index=tesdex)
sub.to_csv("boosted_sub.csv",index=True)
print(cv_scores)
print("RUNTIME %0.2f"%((time.time() - start)/60))