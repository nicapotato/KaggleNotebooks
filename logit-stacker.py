# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy  as np
from scipy.special import expit, logit
 
almost_zero = 1e-10
almost_one  = 1-almost_zero

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

scores = {}

id_col = "click_id"
target = "is_attributed"

df = pd.read_csv("../input/simple-averaging/submission_final.csv",index_col=id_col).rename(columns={target: '1'}) # 0.80121
scores["1"] = 0.95
df["2"] = pd.read_csv("../input/notebook-version-of-talkingdata-lb-0-9786/sub_it24.csv")[target].values # 0.793
scores["2"] = 0.80
df["3"] = pd.read_csv("../input/log-and-harmonic-mean-lets-go/submission_geo.csv")[target].values # 0.78583
scores["3"] = 0.50

# More NN..
# Add https://www.kaggle.com/emotionevil/nlp-and-stacking-starter-dpcnn-lgb-lb0-80/notebook

weights = [0] * len(df.columns)
power = 120
dic = {}

for i,col in enumerate(df.columns):
    weights[i] = scores[col] ** power
    dic[i] = df[col].clip(almost_zero,almost_one).apply(logit) * weights[i] 
    
print(weights[:])
totalweight = sum(weights)

temp = []
for x in dic:
    if x == 0:
        temp = dic[x]
    else:
        temp = temp+dic[x]

# Average
temp = temp/(totalweight)

df[target] = temp
df[target] = df[target].apply(expit)
df[target].to_csv("ensembling_submission.csv", index=True, header=True)
print(df[target].head())