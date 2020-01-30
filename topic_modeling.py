import pandas as pd
import gensim
import matplotlib.pyplot as plt
from data_cleaning import tokenize_lemma_stem
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


def print_most_important_topics_in_seat_reviews(topic_model, num_topics):
    full_comment_df = pd.read_csv('Scraped_data/Skytrax/skytrax_reviews_data.csv', index_col=None)
    full_comment_df['comment_id'] = full_comment_df.index
    seat_review_ids = full_comment_df[full_comment_df.review_type == 'Seat Reviews'].index
    df_test = full_comment_df[full_comment_df.comment_id.isin(list(seat_review_ids))]
    df_test['model_topics'] = df_test.bow_corpus.apply(topic_model.get_document_topics)

    topic_importance = dict.fromkeys(range(num_topics), 0)
    for model_topic in df_test.model_topics:
        for topic in model_topic:
            topic_importance[topic[0]] += topic[1]

    print('Most important topic in Seat Reviews :\n{}'.format(
        {k: v for k, v in sorted(topic_importance.items(), key=lambda item: item[1], reverse=True)}))


sentences_file_path = 'noun_sentences_for_topic_modeling.csv'
data_output_path = 'data_with_topics.csv'


if __name__ == '__main__':
    df = pd.read_csv(sentences_file_path)
    try:
        df['comment_token_stem_lemma'] = df.noun.apply(tokenize_lemma_stem)
    except AttributeError:
        df['comment_token_stem_lemma'] = df.sentence.apply(tokenize_lemma_stem)

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
    dictionary.filter_extremes(no_below=50, no_above=0.30, keep_n=100000)
    dictionary.save('Models/dictionary')
    print("Dictionary saved")

    num_topics = 11

    # # Run LDA using Bag of Words
    df['bow_corpus'] = df.comment_token_stem_lemma.apply(lambda doc: dictionary.doc2bow(doc))
    for num_topics in range(5, 20):
        lda_model_bag_of_words = gensim.models.LdaMulticore(df.bow_corpus, num_topics=num_topics, id2word=dictionary,
                                                            passes=2, workers=4)
        print('Topics using Bag-Of-Words with {} topics'.format(num_topics))
        for sentence_index, topic in lda_model_bag_of_words.print_topics(-1):
            print('Topic: {} Word: {}'.format(sentence_index, topic))
        lda_model_bag_of_words.save('Models/LDA_bag_of_words_{}_topics'.format(num_topics))
        print("LDA model with Bag of Words saved")

    # Run LDA using TF-IDF
    tfidf = gensim.models.TfidfModel(list(df.bow_corpus))
    corpus_tfidf = tfidf[list(df.bow_corpus)]
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=num_topics, id2word=dictionary, passes=2,
                                                 workers=4)

    print('Topics using TF-IDF')
    for idx, topic in lda_model_tfidf.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic))

    # # Tune number of topics hyperparameter with coherence metric (long to run)
    # start, stop, step = 20, 30, 1
    # models, coherence_values = get_coherence_values(df.bow_corpus, df.comment_token_stem_lemma, dictionary)
    # plot_coherence_values(coherence_values, start, stop, step)
    # best_models = sorted(list(zip(models, coherence_values)), key=operator.itemgetter(1), reverse=True)

    # Test topic modeling by predicting topics on 'Seat Reviews of Skytrax which are aircraft interior related
    print_most_important_topics_in_seat_reviews(lda_model_bag_of_words, num_topics)
    lda_model_bag_of_words.save('Models/lda_model_bag_of_words')

    # Classify sentences by topic
    lda_model_bag_of_words = gensim.models.LdaMulticore.load('lda_model')

    df['sentence_topic'] = df.bow_corpus.apply(lda_model_bag_of_words.get_document_topics)
    df.to_csv('sentences_classified.csv')
