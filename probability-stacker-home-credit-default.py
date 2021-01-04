# Adapted From Serigne's work:
# https://www.kaggle.com/serigne/top-8-with-geometric-mean
# https://www.kaggle.com/serigne/lb-0-286-with-harmonic-mean/code

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from scipy.stats.mstats import gmean
from scipy.stats.mstats import hmean

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

target_name = "TARGET"
index_name = "SK_ID_CURR"

# Any results you write to the current directory are saved as output.
subs = pd.read_csv('../input/fork-lightgbm-with-simple-features/submission_kernel02.csv').rename(columns = {target_name:"james"})
subs["kxx"] = pd.read_csv("../input/tidy-xgb-all-tables-0-789/tidy_xgb_0.78801.csv")[target_name]
subs["A"] = pd.read_csv('../input/updated-0-792-lb-lightgbm-with-simple-features/submission_kernel02.csv')[target_name]
subs.set_index(index_name, inplace=True)

subs.head()
# HYPER PARAMETERS
scores ={
    'james': .788,
     'kxx': .789,
     'A': .792,
     #'O2': .795
             }
A_mean_weights = [.4,.3,.3]
gweights = [.6,.2,.2]

# Logic
power = 400

# Systems
# Harmonic Mean doesn't like zeroes
subs = subs.replace(0,0.000001)
# Apply Geometric Mean 
geo_mean = subs.apply(gmean,axis=1).rename(target_name)
geo_mean.to_csv('sub_geometric.csv', index = True, header=True)
print(geo_mean.head())

# Weighted Geometric Mean
import math
sum([x*y for x,y in zip()])

#Apply harmonic mean 
h_mean = subs.apply(hmean,axis=1).rename(target_name)
h_mean.to_csv('sub_harmonic.csv', index = True, header=True)
print(h_mean.head())

# Old School Shenanigans
blend = pd.Series((subs.iloc[:,0] * A_mean_weights[0])
    + (subs.iloc[:,1] * A_mean_weights[1])
    + (subs.iloc[:,2] * A_mean_weights[2])
    #+(subs.iloc[:,2] * A_mean_weights[3])
    ).rename(target_name)
blend.to_csv('weighted_mean.csv', index=True, header=True)
print(blend.head())

# Logit Blend
from scipy.special import expit, logit
almost_zero = 1e-10
almost_one  = 1-almost_zero
weights = [0] * len(subs.columns)
dic = {}

subs.columns
# ROC Scores
for i,col in enumerate(subs.columns):
    weights[i] = scores[col] ** power
    dic[i] = subs[col].clip(almost_zero,almost_one).apply(logit) * weights[i] 
    
print(weights[:])
totalweight = sum(weights)

temp = []
for x in dic:
    if x == 0:
        temp = dic[x]
    else:
        temp = temp+dic[x]

logit_blend = pd.Series((temp/(totalweight)).apply(expit)).rename(target_name)
logit_blend.to_csv('logit_blend.csv',index=True,header=True)
print(logit_blend.head())