import pandas as pd
import gensim
import matplotlib.pyplot as plt
from NLP_Interior_Airline_Services.data_cleaning import tokenize_lemma_stem


def get_coherence_values(corpus, texts, dictionary, start=11, stop=18, step=1):
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

# num_topics = 15
# # Run LDA using bag of words
# lda_model = gensim.models.LdaMulticore(df.bow_corpus, num_topics=num_topics, id2word=dictionary, passes=2, workers=4)
# for idx, topic in lda_model.print_topics(-1):
#     print('Topic: {} \nWords: {}'.format(idx, topic))

# # Run LDA using TF-IDF
# lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=num_topics, id2word=dictionary, passes=2, workers=4)
# for idx, topic in lda_model_tfidf.print_topics(-1):
#     print('Topic: {} Word: {}'.format(idx, topic))

# Tune number of topics hyperparameter with coherence metric (long to run)
models, coherence_values = get_coherence_values(corpus_tfidf, df.tokenized_sentences, dictionary)


# Pick the interesting topics and test if they are preponderent in Seat Reviews:
# Topics 0, 2, 9
full_comment_df = pd.read_csv('skytrax_reviews_data.csv')
seat_review_ids = full_comment_df[full_comment_df.review_type == 'Seat Reviews'].index

df_test = df[df.comment_id.isin(list(seat_review_ids))]
df_test['model_topics'] = df_test.bow_corpus.apply(lda_model.get_document_topics)

topic_importance = dict.fromkeys(range(num_topics), 0)
for model_topic in df_test.model_topics:
    for topic in model_topic:
        topic_importance[topic[0]] += topic[1]

print('Most important topic in Seat Reviews :\n{}'.format(
    {k: v for k, v in sorted(topic_importance.items(), key=lambda item: item[1], reverse=True)}))
