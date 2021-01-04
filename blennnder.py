# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy  as np
from scipy.special import expit, logit
import gc
 
almost_zero = 1e-10
almost_one  = 1-almost_zero

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

scores = {}


df = pd.read_csv("../input/psst-wanna-blend-some-more-and-more/average_result.csv", index_col="click_id",encoding='utf-8').rename(columns={'is_attributed': 'psst'})
scores["psst"] = 0.9689
testdex = df.index

# # MINE
df["mine"] = pd.read_csv("../input/alexey-s-lightgbm-nick-mod-0-9681/submission.csv", encoding='utf-8')['is_attributed'].values # 0.78583
scores["mine"] = 0.9681 *1.1


df["avg_rank"] = pd.read_csv("../input/rank-averaging-on-talkingdata/rank_averaged_submission.csv", encoding='utf-8')['is_attributed'].values # 0.78583
scores["avg_rank"] = 0.9697

# R LIGHT GBM
df["RLGBM"] = pd.read_csv("../input/single-lightgbm-in-r-with-75-mln-rows-lb-0-9687/sub_lightgbm_R_75m.csv", encoding='utf-8')['is_attributed'].values # 0.7959
scores["RLGBM"] = 0.9683

print("Done Read")
weights = [0] * len(df.columns)
power = 120
dic = {}

for i,col in enumerate(df.columns):
    weights[i] = scores[col] ** power
    dic[i] = df.loc[:,col].clip(almost_zero,almost_one).apply(logit) * weights[i] 

del df
gc.collect

print(weights[:])
totalweight = sum(weights)

temp = dic[0] + dic[1] + dic[2] #+ dic[3]
# for x in dic:
#     if x == 0:
#         temp = dic[x]
#     else:
#         temp = temp+dic[x]

del dic
gc.collect

print("Done Addition")
temp = temp/(totalweight)

sub = pd.DataFrame(index=testdex)#.set_index(testdex).rename(columns={0:"is_attributed"})
sub["is_attributed"] = temp

del temp
gc.collect

sub.head()
sub.columns
sub = sub['is_attributed'].apply(expit)
print("TO CSV")
sub.to_csv("ensembling_submission.csv", index=True,header=True)
print(sub.head())
