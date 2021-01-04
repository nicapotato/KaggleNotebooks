# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Sentiment Build
SIA = SentimentIntensityAnalyzer()

df = pd.read_csv(
    '../input/jigsaw-toxic-comment-classification-challenge/train.csv', index_col="id").fillna(' ')#.sample(1000)
test = pd.read_csv(
    '../input/jigsaw-toxic-comment-classification-challenge/test.csv', index_col="id").fillna(' ')#.sample(1000)
badwords = pd.read_csv(
    '../input/bad-bad-words/bad-words.csv', header=None).iloc[:,0].tolist()
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train_text = df['comment_text']
test_text = test['comment_text']
all_text = pd.DataFrame(pd.concat([train_text, test_text]))

print("Creating Custom DENSE Features..")
def eng1(df):
    """
    SOURCE: https://www.kaggle.com/eikedehling/feature-engineering
    """
    # Length of the comment - my initial assumption is that angry people write short messages
    df['num_chars'] = df['comment_text'].apply(len)
    # Number of capitals - observation was many toxic comments being ALL CAPS
    df['capitals'] = df['comment_text'].apply(
        lambda comment:sum(1 for c in comment if c.isupper()))
    # Number of exclamation marks - i observed several toxic comments with multiple exclamation marks
    df['num_exclamation_marks'] = df['comment_text'].apply(
        lambda comment: comment.count('!'))
    # Number of question marks - assumption that angry people might not use question marks
    df['num_question_marks'] = df['comment_text'].apply(
        lambda comment: comment.count('?'))
        
    # Number of punctuation symbols - assumption that angry people might not use punctuation
    df['num_punctuation'] = df['comment_text'].apply(
        lambda comment: sum(comment.count(w) for w in '.,;:'))
    # Number of symbols - assumtion that words like fck or $# or sh*t mean more symbols in foul language (Thx for tip!)
    df['num_symbols'] = df['comment_text'].apply(
        lambda comment: sum(comment.count(w) for w in '*&$%'))
    # Number of words - angry people might write short messages?
    df['num_words'] = df['comment_text'].apply(
        lambda comment: len(comment.split()))
    # LOWER CASE
    df["comment_text"] = df["comment_text"].str.lower()
    # Number of unique words - observation that angry comments are sometimes repeated many times
    df['num_unique_words'] = df['comment_text'].apply(
        lambda comment: len(set(w for w in comment.split())))
    
    # Number of (happy) smilies - Angry people wouldn't use happy smilies, right?
    df['num_smilies'] = df['comment_text'].apply(
        lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))
    # Proportion of unique words - see previous
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']*100
    df["badwordcount"] = df['comment_text'].apply(
    lambda comment: sum(comment.count(w) for w in badwords))
    return df

def bad_words(df):
    print("part 3")
    df["normchar_badwords"] = (df["badwordcount"]/df['num_chars'])*100
    df["normword_badwords"] = (df["badwordcount"]/df['num_words'])*100
    # Proportion of capitals - see previous
    df['caps_vs_length'] = df.apply(
        lambda row: float(row['capitals'])/float(row['num_chars'])*100,axis=1)
    return df

def sentiment_analysis(df):
    # Pre-Processing
    df["nltk_vader_Compound Score"]= df['comment_text'].apply(lambda x:SIA.polarity_scores(x)['compound'])
    df['nltk_vader_Neutral Score']= df['comment_text'].apply(lambda x:SIA.polarity_scores(x)['neu'])
    df['nltk_vader_Negative Score']= df['comment_text'].apply(lambda x:SIA.polarity_scores(x)['neg'])
    df['nltk_vader_Positive Score']= df['comment_text'].apply(lambda x:SIA.polarity_scores(x)['pos'])
    return df

# Execute Custom Features
new_features = sentiment_analysis(bad_words(eng1(df=all_text)))
new_features = new_features.drop("comment_text",axis=1)

import seaborn as sns
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize= [10,7])
sns.heatmap(new_features.corr(), cmap= plt.cm.plasma,annot=True, fmt=".2f", ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
ax.set_title("ALL DATA- Correlation Between New Features")
plt.savefig('ALL_DATA_Heatmap.png',bbox_inches='tight')

f, ax = plt.subplots(figsize= [14,11])
sns.heatmap(pd.concat([df, new_features.loc[df.index, :]], axis=1).corr(), cmap= plt.cm.plasma, annot=True, fmt=".2f", ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
ax.set_title("TRAIN DATA- Correlation Between New Features and Toxicity")
plt.savefig('TRAIN_DATA_Heatmap.png',bbox_inches='tight')

print("Done")

new_features.to_csv("new_features.csv",index=True)
# used train.index, test.index to extract respective data.