import pandas as pd
from data_cleaning import create_sentence_dataframe, clean_comment_string, tokenize_lemma_stem
import gensim
from operator import itemgetter
import numpy as np
from sentiment_analysis import text_fit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from data_cleaning import tokenize_lemma_stem


# Load test set
test_set = pd.read_table('test_data.csv', encoding='latin_1')
test_set.columns = ['comment']
test_set['comment_id'] = test_set.index

# Clean the comment string
test_set.comment = test_set.comment.apply(lambda x: x.lower())
test_set.comment = test_set.comment.apply(clean_comment_string)

# Load pre-trained topic modeling dictionary and model
lda_model = gensim.models.LdaMulticore.load('Models/LDA_bag_of_words_{}_topics')
dictionary = gensim.corpora.Dictionary.load('Models/dictionary')

sentence_dataframe = create_sentence_dataframe(test_set, filter_nouns=True)
sentence_dataframe['tokens'] = sentence_dataframe.sentence.apply(tokenize_lemma_stem)
sentence_dataframe['bow_corpus'] = sentence_dataframe.tokens.apply(lambda x: dictionary.doc2bow(x))
sentence_dataframe['topics'] = sentence_dataframe.bow_corpus.apply(lda_model.get_document_topics)
sentence_dataframe['most_likely_topic'] = sentence_dataframe.topics.apply(lambda x: max(x, key=itemgetter(1))[0])


topics = {0: 'crew_food_and_servings',
          1: 'check_in_and_delays',
          2: 'classes',
          3: 'comfort_and_extras',
          4: 'entertainment'}

# Join back the sentences with topics to original dataframe
split_sentences_dictionary = dict.fromkeys(sentence_dataframe.comment_id.unique())
for comment_id in split_sentences_dictionary.keys():
    split_sentences_dictionary[comment_id] = dict.fromkeys(topics.values())
    for topic in topics.values():
        split_sentences_dictionary[comment_id][topic] = []

for _, row in sentence_dataframe.iterrows():
    split_sentences_dictionary[row.comment_id][topics[row.most_likely_topic]].append(row.sentence)

topic_df = pd.DataFrame(split_sentences_dictionary).transpose()

# Join sentences
for topic in topics.values():
    topic_df[topic] = topic_df[topic].apply(lambda x: ' '.join(x))
    topic_df[topic + '_sentiment'] = np.nan
    topic_df = topic_df.rename(columns={topic: topic + '_sentences'})


test_set = test_set.join(topic_df)
test_set = test_set.drop(['comment_id'], axis=1)
test_set = test_set.replace('', np.nan)

# Re-train sentiment analysis model
train_set = pd.read_csv('Scraped_data/Skytrax/skytrax_reviews_data.csv').dropna(subset=['rating', 'comment'])

tf_idf = TfidfVectorizer(stop_words='english')
model_preprocessing, model_sentiment = text_fit(train_set['comment'], train_set['rating'], tf_idf, LogisticRegression())

for topic in topics.values():
    test_set[topic + '_sentences_preprocessed'] = test_set[topic + '_sentences'].apply(lambda x: model_preprocessing.transform([x]) if type(x) == str else np.nan)
    test_set[topic + '_sentiment'] = test_set[topic + '_sentences_preprocessed'].apply(lambda x: model_sentiment.predict_proba(x)[0][1] if type(x) != float else np.nan)




test_set = test_set.dropna(axis=1, how='all')
test_set = test_set.loc[:, ['preprocessed' not in col for col in test_set.columns]]

def remove_unknown_words():
    pass

