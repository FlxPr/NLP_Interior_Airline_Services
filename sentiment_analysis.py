
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
# nltk.download('punkt')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from string import punctuation
from sklearn import svm
from nltk.corpus import stopwords

from fractions import Fraction
import re


from sklearn.dummy import DummyClassifier
from string import punctuation
from sklearn import svm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from nltk import ngrams
from itertools import chain

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb


df = pd.read_csv('data_with_topics.csv')
df.head()

# drop the rows having null values for reviews text
df = df.dropna(subset=['comment'])
df = df.dropna(subset=['rating'])

# Mapping the ratings into 0 and 1 which is negative and positive respectively
X = df['comment']
y_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}
y = df['rating'].map(y_dict)

c = CountVectorizer(stop_words='english')


def text_fit(x, y, model, clf_model, coef_show=1):
    x_c = model.fit_transform(x)
    print('# features: {}'.format(x_c.shape[1]))
    x_train, x_test, y_train, y_test = train_test_split(x_c, y, random_state=0)
    print('# train records: {}'.format(x_train.shape[0]))
    print('# test records: {}'.format(x_test.shape[0]))
    clf = clf_model.fit(x_train, y_train)
    acc = clf.score(x_test, y_test)
    print('Model Accuracy: {}'.format(acc))

    if coef_show == 1:
        w = model.get_feature_names()
        coef = clf.coef_.tolist()[0]
        coeff_df = pd.DataFrame({'Word': w, 'Coefficient': coef})
        coeff_df = coeff_df.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
        print('')
        print('-Top 20 positive-')
        print(coeff_df.head(20).to_string(index=False))
        print('')
        print('-Top 20 negative-')
        print(coeff_df.tail(20).to_string(index=False))
    return model, clf


# Baseline accuracy of the model
model_preprocessing, model_sentiment = text_fit(X, y, c, LogisticRegression())


# TF-IDF vectorizer is added to logistic regression to improve the model accuracy

tfidf = TfidfVectorizer(stop_words='english')
fitted_tfidf, model_base_logistic = text_fit(X, y, tfidf, LogisticRegression())



# TFIDF + n-grams.
tfidf_n = TfidfVectorizer(ngram_range=(1, 5), stop_words='english')
model_base_logistic_n_grams = text_fit(X, y, tfidf_n, LogisticRegression())



# Airline behavior
df_user = df.groupby(['airline']).agg({'rating': ['count', 'mean']})
df_user.columns = df_user.columns.get_level_values(1)
df_user.columns = ['Rating count', 'Rating mean']
df_user = df_user.sort_values(by='Rating count', ascending=False)
print(df_user.head(10))


# %%

# only the adjectives are taken into consideration, as those express opinion and nouns don't.
def get_token_adj(score, benchmark, airline='all'):
    if airline != 'all':
        DF = df[(df['airline'] == airline) & (df['rating'] == score)]['comment']
    else:
        DF = df[df['rating'] == score]['comment']

    count = len(DF)
    total_text = ' '.join(DF)
    total_text = total_text.lower()
    stop = stopwords
    total_text = nltk.word_tokenize(total_text)
    total_text = [word for word in total_text if word not in stop and len(word) >= 3]
    lemmatizer = WordNetLemmatizer()
    total_text = [lemmatizer.lemmatize(w, 'a') for w in total_text]
    # get adjective only
    total_text = [word for word, form in nltk.pos_tag(total_text) if form == 'JJ']

    text = nltk.Text(total_text)
    fdist = nltk.FreqDist(text)

    # return only phrase occurs more than benchmark of his reviews
    return sorted([(w, fdist[w], str(round(fdist[w] / count * 100, 2)) + '%') for w in set(text) if
                   fdist[w] >= count * benchmark], key=lambda x: -x[1])


# %%

index = ['Phrase', 'Count', 'Occur %']

for j in range(1, 6):
    test = pd.DataFrame()
    d = get_token_adj(j, 0.25, 'Lufthansa')
    print('score {} reviews most popular adjectives word:'.format(j))
    for i in d:
        test = test.append(pd.Series(i, index=index), ignore_index=True)
    test = test.sort_values('Count', ascending=False)
    print(test)


# %%

# Most popular words used by the user for different ratings are observed. (2-grams and 3-grams are chosen for analysis)
def get_token_ngram(score, benchmark, airline='all'):
    if airline != 'all':
        DF = df[(df['airline'] == airline) & (df['rating'] == score)]['comment']
    else:
        DF = df[df['rating'] == score]['comment']

    count = len(DF)
    total_text = ' '.join(DF)
    total_text = total_text.lower()
    stop = stopwords
    total_text = nltk.word_tokenize(total_text)
    total_text = [word for word in total_text if word not in stop and len(word) >= 3]
    lemmatizer = WordNetLemmatizer()
    total_text = [lemmatizer.lemmatize(w, 'v') for w in total_text]
    bigrams = ngrams(total_text, 2)
    trigrams = ngrams(total_text, 3)

    # look at 2-gram and 3-gram together
    combine = chain(bigrams, trigrams)
    text = nltk.Text(combine)
    fdist = nltk.FreqDist(text)

    # return only phrase occurs more than benchmark of his reviews
    return sorted([(w, fdist[w], str(round(fdist[w] / count * 100, 2)) + '%') for w in set(text) if
                   fdist[w] >= count * benchmark], key=lambda x: -x[1])


# score 1-5 reviews with this user
index = ['Phrase', 'Count', 'Occur %']

for j in range(1, 6):
    test = pd.DataFrame()
    d = get_token_ngram(j, 0.25, 'Emirates')
    print('score {} reviews most popular 2-gram / 3-gram:'.format(j))
    for i in d:
        test = test.append(pd.Series(i, index=index), ignore_index=True)
    # test = test.sort_values('Count', ascending=False)
    print(test)

# %%

# score 1-5 reviews with all users
index = ['Phrase', 'Count', 'Occur %']

for j in range(1, 6):
    test = pd.DataFrame()
    # easier benchmark since we have many different users here, thus different phrase
    d = get_token_ngram(j, 0.03)
    print('score {} reviews most popular 2-gram / 3-gram:'.format(j))
    for i in d:
        test = test.append(pd.Series(i, index=index), ignore_index=True)
    test = test.sort_values('Count', ascending=False)
    print(test)

# %%


# %%


# %%

import pandas as pd


df_test = df.comment[:10]

def compute_sentiment(comment, sentiment_model, preprocessing):

    preprocessed_comment = preprocessing.transform([comment])
    return sentiment_model.predict_proba(preprocessed_comment)[0][1]


compute_sentiment('comment good', model_sentiment, model_preprocessing)

