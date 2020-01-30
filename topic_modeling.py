import pandas as pd
import gensim
import matplotlib.pyplot as plt
from NLP_Interior_Airline_Services.data_cleaning import tokenize_lemma_stem


def get_coherence_values(corpus, texts, dictionary, start=4, stop=10, step=1):
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        model = gensim.models.LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = gensim.models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


def plot_coherence_values(coherence_values, start, stop, step):
    plt.plot(range(start, stop, step), coherence_values)
    plt.title('coherence values for different number of topics')
    plt.grid()
    plt.show()


df = pd.read_csv('sentences_for_topic_modelling.csv')
df['tokenized_sentences'] = df.sentence.apply(tokenize_lemma_stem)

# bigram = gensim.models.Phrases(df['tokenized_sentences'])
# texts = [bigram[sentence] for sentence in df['tokenized_sentences']]

# TOPIC MODEL --
dictionary = gensim.corpora.Dictionary(df['tokenized_sentences'])
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
df['bow_corpus'] = df.tokenized_sentences.apply(lambda doc: dictionary.doc2bow(doc))

tfidf = gensim.models.TfidfModel(list(df.bow_corpus))
corpus_tfidf = tfidf[list(df.bow_corpus)]


# PICK THE BEST MODEL ACCORDING TO ITS COHERENCE VALUE
# model_list, coherence_values = get_coherence_values(dictionary, corpus_tfidf, dictionary)

num_topics = 13

# Run LDA using TF-IDF
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=num_topics, id2word=dictionary, passes=2, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))

lda_model_tfidf.save('lda_model')
# topic 0 : Seat comfort and recline
# topic 7 : Seat comfort and extras
# Topic 12 : Entertainment and food


# Tune number of topics hyperparameter with coherence metric (long to run)
models, coherence_values = get_coherence_values(corpus_tfidf, df.tokenized_sentences, dictionary)


# Pick the interesting topics and test if they are preponderent in Seat Reviews:
# Topics 0, 2, 9
full_comment_df = pd.read_csv('skytrax_reviews_data.csv', index_col=None)
seat_review_ids = full_comment_df[full_comment_df.review_type == 'Seat Reviews'].index

df_test = df[df.comment_id.isin(list(seat_review_ids))]
df_test['model_topics'] = df_test.bow_corpus.apply(lda_model_tfidf.get_document_topics)

topic_importance = dict.fromkeys(range(num_topics), 0)
for model_topic in df_test.model_topics:
    for topic in model_topic:
        topic_importance[topic[0]] += topic[1]

print('Most important topic in Seat Reviews :\n{}'.format(
    {k: v for k, v in sorted(topic_importance.items(), key=lambda item: item[1], reverse=True)}))

# Classify sentences by topic
import operator

lda_model_tfidf = gensim.models.LdaMulticore.load('lda_model')
interesting_topics = [0, 7, 12]

df['sentence_topics'] = df.bow_corpus.apply(lda_model_tfidf.get_document_topics)
df['sentence_best_topic'] = df.sentence_topics.apply(lambda x: max(x, key=operator.itemgetter(1))[0])
df.to_csv('sentences_classified.csv')


import pandas as pd
df = pd.read_csv('sentences_classified.csv')
interesting_topics = [0, 7, 12]

d = dict.fromkeys(df.comment_id.unique())
for comment_id in d.keys():
    d[comment_id] = dict.fromkeys(interesting_topics)
    for topic in interesting_topics:
        d[comment_id][topic] = []

count = 0
for _, row in df.iterrows():
    if row.sentence_best_topic in interesting_topics:
        d[row.comment_id][row.sentence_best_topic].append(row.sentence)
    count+=1
    if count%1000 == 0:
        print(count)

topic_df = pd.DataFrame(d).transpose()
topic_names = {0: 'topic_comfort', 7: 'topic_extras', 12: 'topic_entertainment'}
full_comment_df = full_comment_df.join(topic_df)
full_comment_df = full_comment_df.rename(columns=topic_names)
full_comment_df.to_csv('data_with_topics.csv')
