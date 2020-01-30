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
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from data_cleaning import tokenize_lemma_stem
import xgboost as xgb


def text_fit(comments, ratings, tf_idf_model, classifier_model, coef_show=1):
    # comments = comments.apply(lambda x: ''.join(tokenize_lemma_stem(x, pos='')))
    x_tf_idf = tf_idf_model.fit_transform(comments)
    print('# features: {}'.format(x_tf_idf.shape[1]))
    x_train, x_test, y_train, y_test = train_test_split(x_tf_idf, ratings, random_state=0)
    print('# train records: {}'.format(x_train.shape[0]))
    print('# test records: {}'.format(x_test.shape[0]))
    clf = classifier_model.fit(x_train, y_train)
    acc = clf.score(x_test, y_test)
    print('Model Accuracy: {}'.format(acc))

    if coef_show == 1:
        w = tf_idf_model.get_feature_names()
        coef = clf.coef_.tolist()[0]
        coeff_df = pd.DataFrame({'Word': w, 'Coefficient': coef})
        coeff_df = coeff_df.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
    return tf_idf_model, clf


def compute_sentiment(comment, sentiment_model, preprocessing):
    if type(comment) == str:
        preprocessed_comment = preprocessing.transform([comment])
        return sentiment_model.predict_proba(preprocessed_comment)[0][1]
    return np.nan


if __name__ == '__main__':
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

    # Baseline accuracy of the model
    model_preprocessing, model_sentiment = text_fit(X, y, c, LogisticRegression())

    # TF-IDF vectorizer is added to logistic regression to improve the model accuracy
    tfidf = TfidfVectorizer(stop_words='english')
    fitted_tfidf, model_base_logistic = text_fit(X, y, tfidf, LogisticRegression())

    # TFIDF + n-grams.
    tfidf_n = TfidfVectorizer(ngram_range=(1, 5), stop_words='english')
    model_base_logistic_n_grams = text_fit(X, y, tfidf_n, LogisticRegression())
    df_test = df.comment[:10]

    compute_sentiment('comment good', model_sentiment, model_preprocessing)

