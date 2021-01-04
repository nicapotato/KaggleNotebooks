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

# Target Name
target_name = "deal_probability"

# Any results you write to the current directory are saved as output.
subs = pd.read_csv('../input/bow-meta-text-and-dense-features-lgbm/lgsub.csv').rename(columns = {target_name:"LGBM"})
subs["agg_lightGBM"] = pd.read_csv("../input/best-public-blend-0-2204/best_public_blend.csv")[target_name]
subs["xgb"] = pd.read_csv('../input/xgb-text2vec-tfidf-0-2237/xgb_tfidf0.218538.csv')[target_name]
#subs["moyen_push_median"] = pd.read_csv('../input/stacking-mean-minmax-median/stack_pushout_median.csv')[target_name]
subs.set_index("item_id", inplace=True)

# HYPER-PARAMETERS
scores ={'LGBM':.5,
         'agg_lightGBM':.9,
         'xgb':.5,
         #'moyen_push_median': .95
             }
A_mean_weights = [.05,.9,.05]
gweights = [.6,.2,.2]

# Ensembling GO
# Harmonic Mean doesn't like zeroes
subs = subs.replace(0,0.000001)

# Apply Geometric Mean 
geo_mean = subs.apply(gmean,axis=1).rename(target_name)
geo_mean.to_csv('sub_geometric.csv', index = True, header=True)
print(geo_mean.head())

# Weighted Geometric Mean
import math
gweights = [.6,.2,.2]
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
blend.to_csv('weighted_mean.csv', index=True)
print(blend.head())

# Logit Blend
from scipy.special import expit, logit
almost_zero = 1e-10
almost_one  = 1-almost_zero
weights = [0] * len(subs.columns)
power = 10
dic = {}

print(subs.columns)
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