import pandas as pd
import gensim
import matplotlib.pyplot as plt
from NLP_Interior_Airline_Services.data_cleaning import tokenize_lemma_stem
import operator
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from nltk.corpus import stopwords


def get_coherence_values(corpus, texts, dictionary, start=4, stop=20, step=1):
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        model = gensim.models.LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = gensim.models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    plot_coherence_values(coherence_values, start, stop, step)
    return model_list, coherence_values


def plot_coherence_values(coherence_values, start, stop, step):
    plt.plot(range(start, stop, step), coherence_values)
    plt.title('Coherence values for different number of topics')
    plt.xlabel('Number of topics')
    plt.ylabel('Coherence score')
    plt.grid()
    plt.show()


def print_most_important_topics_in_seat_reviews(topic_model):
    full_comment_df = pd.read_csv('skytrax_reviews_data.csv', index_col=None)
    seat_review_ids = full_comment_df[full_comment_df.review_type == 'Seat Reviews'].index
    df_test = df[df.comment_id.isin(list(seat_review_ids))]
    df_test['model_topics'] = df_test.bow_corpus.apply(topic_model.get_document_topics)

    topic_importance = dict.fromkeys(range(num_topics), 0)
    for model_topic in df_test.model_topics:
        for topic in model_topic:
            topic_importance[topic[0]] += topic[1]

    print('Most important topic in Seat Reviews :\n{}'.format(
        {k: v for k, v in sorted(topic_importance.items(), key=lambda item: item[1], reverse=True)}))


def plot_wordcloud(): # TODO refactor because code taken from internet or delete
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
    topics = lda_model.show_topics(10, formatted=False)

    for idx, topic in lda_model_bag_of_words.print_topics(-1):
        print(type(topic))
        print('Topic: {} Word: {}'.format(idx, topic))

    for i in range(len(topics)):
        plt.figure()
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    df = pd.read_csv('noun_sentences_for_topic_modelling.csv')
    df['comment_token_stem_lemma'] = df.noun.apply(tokenize_lemma_stem)
    df = df.dropna(subset=['comment_token_stem_lemma'])

    # Create bigrams and trigrams
    bigram = gensim.models.Phrases(df['comment_token_stem_lemma'], min_count=30)
    trigram = gensim.models.Phrases(bigram[df['comment_token_stem_lemma']], min_count=20)
    for sentence_index in range(len(df.index)):
        for token in bigram[df['comment_token_stem_lemma'].iloc[sentence_index]]:
            if token.count('_') == 1:
                df['comment_token_stem_lemma'].iloc[sentence_index].append(token)
        for token in trigram[df['comment_token_stem_lemma'].iloc[sentence_index]]:
            if token.count('_') == 2:
                df['comment_token_stem_lemma'].iloc[sentence_index].append(token)

    # TOPIC MODEL - Create dictionary
    dictionary = gensim.corpora.Dictionary(df['comment_token_stem_lemma'])
    dictionary.filter_extremes(no_below=50, no_above=0.20, keep_n=100000)

    num_topics = 10

    # # Run LDA using Bag of Words
    df['bow_corpus'] = df.comment_token_stem_lemma.apply(lambda doc: dictionary.doc2bow(doc))
    lda_model_bag_of_words = gensim.models.LdaMulticore(df['bow_corpus'], num_topics=num_topics, id2word=dictionary, passes=2, workers=4)
    for sentence_index, topic in lda_model_bag_of_words.print_topics(-1):
       print('Topic: {} Word: {}'.format(sentence_index, topic))

    # Run LDA using TF-IDF
    tfidf = gensim.models.TfidfModel(list(df.bow_corpus))
    corpus_tfidf = tfidf[list(df.bow_corpus)]
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=num_topics, id2word=dictionary, passes=2, workers=4)
    for idx, topic in lda_model_tfidf.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic))

    # Tune number of topics hyperparameter with coherence metric (long to run)
    start, stop, step = 20, 30, 1
    models, coherence_values = get_coherence_values(df['bow_corpus'], df.comment_token_stem_lemma, dictionary)
    plot_coherence_values(coherence_values, start, stop, step)
    best_models = sorted(list(zip(models, coherence_values)), key=operator.itemgetter(1), reverse=True)

    # Test topic modeling by predicting topics on 'Seat Reviews of Skytrax which are aircraft interior related
    print_most_important_topics_in_seat_reviews(best_models[0][0])
    lda_model_bag_of_words.save('Models/lda_model_bag_of_words')

    # Classify sentences by topic
    lda_model_bag_of_words = gensim.models.LdaMulticore.load('lda_model')
    interesting_topics = [0, 7, 12]

    df['sentence_topic'] = df.bow_corpus.apply(lda_model_bag_of_words.get_document_topics)
    df.to_csv('sentences_classified.csv')

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
