import pandas as pd
import gensim
import matplotlib.pyplot as plt
from NLP_Interior_Airline_Services.data_cleaning import tokenize_lemma_stem
import operator


def get_coherence_values(corpus, texts, dictionary, start=4, stop=20, step=1):
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

bigram = gensim.models.Phrases(df['tokenized_sentences'], min_count=30)
trigram = gensim.models.Phrases(bigram[df['tokenized_sentences']], min_count=20)


for idx in range(len(df.index)):
    for token in bigram[df['tokenized_sentences'].iloc[idx]]:
        if '_' in token:
           df['tokenized_sentences'].iloc[idx].append(token)
    for token in trigram[df['tokenized_sentences'].iloc[idx]]:
        if token.count('_') == 2:
           df['tokenized_sentences'].iloc[idx].append(token)

# texts = [bigram[sentence] for sentence in df['tokenized_sentences']]

# TOPIC MODEL --
dictionary = gensim.corpora.Dictionary(df['tokenized_sentences'])
dictionary.filter_extremes(no_below=10, no_above=0.10, keep_n=100000)  # TODO check no_above

print('most common tokens: {}'.format(list(dictionary.items())[:10]))
print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(df.index))

df['bow_corpus'] = df.tokenized_sentences.apply(lambda doc: dictionary.doc2bow(doc))

tfidf = gensim.models.TfidfModel(list(df.bow_corpus))
corpus_tfidf = tfidf[list(df.bow_corpus)]

# PICK THE BEST MODEL ACCORDING TO ITS COHERENCE VALUE
# model_list, coherence_values = get_coherence_values(dictionary, corpus_tfidf, dictionary)

num_topics = 13

# # Run LDA using TF-IDF
# lda_model_tfidf = gensim.models.LdaMulticore(df['bow_corpus'], num_topics=num_topics, id2word=dictionary, passes=2, workers=4)
# for idx, topic in lda_model_tfidf.print_topics(-1):
#     print('Topic: {} Word: {}'.format(idx, topic))
#

# # Run LDA using TF-IDF
# lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=num_topics, id2word=dictionary, passes=2, workers=4)
# for idx, topic in lda_model_tfidf.print_topics(-1):
#     print('Topic: {} Word: {}'.format(idx, topic))

# lda_model_tfidf.save('lda_model')
# topic 0 : Seat comfort and recline
# topic 7 : Seat comfort and extras
# Topic 12 : Entertainment and food


# Tune number of topics hyperparameter with coherence metric (long to run)
start = 4
stop = 20
step = 2
models, coherence_values = get_coherence_values(df['bow_corpus'], df.tokenized_sentences, dictionary)

best_models = sorted(list(zip(models, coherence_values)), key=operator.itemgetter(1), reverse=True)
plot_coherence_values(coherence_values, start, stop, step)
# Pick the interesting topics and test if they are preponderent in Seat Reviews:
# Topics 0, 2, 9
full_comment_df = pd.read_csv('skytrax_reviews_data.csv', index_col=None)
seat_review_ids = full_comment_df[full_comment_df.review_type == 'Seat Reviews'].index

df_test = df[df.comment_id.isin(list(seat_review_ids))]
df_test['model_topics'] = df_test.bow_corpus.apply(best_models[0][0].get_document_topics)

topic_importance = dict.fromkeys(range(num_topics), 0)
for model_topic in df_test.model_topics:
    for topic in model_topic:
        topic_importance[topic[0]] += topic[1]

print('Most important topic in Seat Reviews :\n{}'.format(
    {k: v for k, v in sorted(topic_importance.items(), key=lambda item: item[1], reverse=True)}))


# Classify sentences by topic
lda_model_tfidf = gensim.models.LdaMulticore.load('lda_model')
interesting_topics = [0, 7, 12]

def plot_wordcloud():
    # 1. Wordcloud of Top N words in each topic
    from matplotlib import pyplot as plt
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.colors as mcolors
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(
        ['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get',
         'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot',
         'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    lda_model = gensim.models.LdaMulticore.load('lda_model')
    topics = lda_model.show_topics(13, formatted=False)
    topics = [topics[0], topics[7], topics[12]]
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    topic_names = ('Comfort', 'Seat Room & Extras', 'Entertainment & Service')
    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title(topic_names[i], fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()

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
    count += 1
    if count % 1000 == 0:
        print(count)

topic_df = pd.DataFrame(d).transpose()
topic_names = {0: 'topic_comfort', 7: 'topic_extras', 12: 'topic_entertainment'}
full_comment_df = full_comment_df.join(topic_df)
full_comment_df = full_comment_df.rename(columns=topic_names)
full_comment_df.to_csv('data_with_topics.csv')
