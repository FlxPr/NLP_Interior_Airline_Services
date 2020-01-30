import pandas as pd
import numpy as np
import itertools
import re

import gensim
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import nltk.data
import en_core_web_sm


# Uncomment if wordnet and punkt have not been downloaded
# nltk.download('wordnet')
# nltk.download('punkt')

nlp = en_core_web_sm.load()


def clean_aircraft_string(aircraft_string):
    """
    Normalizes aircraft string, eg AIRBUS A380 -> a380
    :param aircraft_string: aircraft string
    :return: cleaned aircraft string
    """
    if type(aircraft_string) == str:
        aircraft_string = aircraft_string.lower()
        aircraft_string = aircraft_string.replace('airbus', '')
        aircraft_string = aircraft_string.replace('and', '/')
        aircraft_string = aircraft_string.replace('embraer ', 'e')
        aircraft_string = aircraft_string.strip()
        return aircraft_string
    return np.nan


def clean_header_string(header_string):
    """
    Cleans the header by removing quote characters and common noisy words
    :param header_string:
    :return: cleaned header string
    """
    if type(header_string) == str:
        header_string = header_string.lower()
        if 'seat review' in header_string or 'customer review' in header_string:
            return np.nan
        header_string = header_string.replace('"', '')
        header_string = header_string.replace('embraer ', 'e')
        header_string = header_string.strip()
        return header_string
    return np.nan


def delete_verified_review_prefix(comment_string):
    """
    Some comments begin by 'VERIFIED REVIEW |'. Removes such prefixes.
    :param comment_string: string containing the review
    :return: cleaned review string
    """
    if '|' in comment_string:
        if 'verified' in comment_string.split('|')[0].lower():
            return comment_string.split('|')[1]
    return comment_string


def reduce_word_exaggeration(comment_string):
    """
    Reduces word exaggeration such as 'veeeeery' to 'veery' which the spellchecker can handle better
    :param comment_string: string containing the review
    :return: cleaned review string
    """
    exaggeration = re.compile(r"(.)\1{2,}")
    return exaggeration.sub(r"\1\1", comment_string)


def remove_characters(comment_string):
    '''
    Removes newline and such characters from the review string
    :param comment_string: string containing the review
    :return: cleaned review string
    '''
    comment_string = re.sub('\s+', ' ', comment_string)
    comment_string = re.sub("\'", "", comment_string)
    return comment_string


def remove_airline_name_from_comment(comment, airline_name):
    '''
    Tries to remove the airline name from the comment and makes other small changes to remove noisy frequent words
    :param comment: string containing the review
    :param airline_name: string containing the airline name
    :return: cleaned review string
    '''
    airline_name_words = airline_name.lower().split(' ')
    for number_of_words in range(1, len(airline_name_words) + 1)[::-1]:
        comment = comment.replace(' '.join(airline_name_words[:number_of_words]), '')
    # further remove common city names and aircraft types
    comment = comment.replace('hong kong', '').replace('london', '').replace('francisco', '').replace('bangkok', '')
    comment = comment.replace('boeing', '').replace('airbus', '').replace('airline', '')
    comment = comment.replace('leg room', 'legroom')
    return comment


def append_header_to_comment(comment, header, twice=True):
    """
    Adds the header to the body of the review to include it in further text processing
    :param comment: string containing the review
    :param header: string containing the header of the review
    :param twice: If True, appends the header twice to give it more weight
    :return:
    """
    if type(header) == str:
        return '{}. {}. {}'.format(header, header, comment) if twice else '{}. {}'.format(header, comment)
    return comment


def clean_comment_string(comment_string):
    """
    Applies several predefined functions to the comment string
    :param comment_string: string containing th ereview
    :return: cleaned review string
    """
    comment_string = delete_verified_review_prefix(comment_string)
    comment_string = reduce_word_exaggeration(comment_string)
    comment_string = remove_characters(comment_string)
    return comment_string


def tokenize_lemma_stem(comment, pos='n'):
    """
    Removes stopwords, preprocesses, lemmatizes and stems the words
    :param comment: string containing the review
    :return: list of stemmed and lemmatized tokens excluding the stop words
    """
    result = []
    if type(comment) == str:
        for token in gensim.utils.simple_preprocess(comment):
            if token not in gensim.parsing.preprocessing.STOPWORDS\
                    and (len(token) > 3
                         or token in ['leg', 'arm', 'eye', 'old', 'bag', 'hip', 'eat',
                                      'row', 'low', 'big', 'age', 'hot', 'kid', 'gap', 'ife']):
             result.append(PorterStemmer().stem(WordNetLemmatizer().lemmatize(token, pos='n')))
        return result
    return np.nan


# def spell_correct(data_frame):  # TODO integrate in tokenizer
#     data_frame = data_frame.copy()
#     spell = SpellChecker()
#     spell.word_frequency.load_words(data_frame.aircraft.dropna().unique())
#     spell.known(['a380'])
#     words = set.union(*[set(comment.split(' ')) for comment in df.comment.str.lower()])


def create_sentence_dataframe(comment_dataframe, filter_nouns=False):
    """
    Splits the reviews by sentence for topic modelling. Keeps track of original comment of the sentence
    :param comment_dataframe: Dataframe containing the cleaned reviews
    :param filter_nouns: If True, applies part-of-speech recognition to keep only the nouns (useful for topic
                            modeling). Time-intensive process.
    :return: Dataframe containing sentences and the corresponding id of the original comment
    """
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    id_sentence_list = []
    for _, comment_row in comment_dataframe.iterrows():
        sentences = tokenizer.tokenize(comment_row['comment'])
        sentences_with_only_nouns = []
        if filter_nouns:
            for i in range(len(sentences)):
                sentences_with_only_nouns.append(' '.join([word.text for word in nlp(sentences[i])
                                                           if word.pos_ == 'NOUN' and len(word) > 2]))

                id_sentence_list.append([comment_row['comment_id'], sentences[i], sentences_with_only_nouns[i]])
        else:
            id_sentence_list.append(zip([comment_row['comment_id']] * len(sentences), sentences))

    id_sentence_dataframe = pd.DataFrame(id_sentence_list, columns=['comment_id', 'sentence', 'nouns']
                                                           if filter_nouns else ['comment_id', 'sentence'])
    return id_sentence_dataframe


def merge_skytrax_tripadvisor_data(df_skytrax, df_tripadvisor):
    """
    Merges the skytrax and tripadvisor scraped data.
    :param df_skytrax: scraped skytrax data
    :param df_tripadvisor: scraped tripadvisor data
    :return: merged dataframe
    """
    dict_columns_to_rename = {'star': 'rating',
                              'title': 'header',
                              'comment': 'comment',
                              'date': 'comment_date',
                              'route': 'Route',
                              'trip_class': 'Seat Type',
                              'Legroom': 'Seat Legroom',
                              'Seat comfort': 'Seat Comfort',
                              'In-flight Entertainment': 'Inflight Entertainment',
                              'Value for money': 'Value For Money',
                              'Food and Beverage': 'Food & Beverages'}

    df_tripadvisor.rename(columns=dict_columns_to_rename, inplace=True)

    df_tripadvisor['Seat Type'].replace(to_replace='Economy', value='Economy Class')
    df_tripadvisor['best_rating'] = 5

    df_tripadvisor['Website'] = 'Tripadvisor'
    df_skytrax['Website'] = 'Skytrax'

    aggregated_df = df_skytrax.append(df_tripadvisor, sort=True)
    return aggregated_df


if __name__ == '__main__':
    df = pd.read_csv('Scraped_data/Skytrax/skytrax_reviews_data.csv').dropna(subset=['comment'])
    #df_tripadvisor = pd.read_csv('Scraped_data/Tripadvisor/tripadvisor_reviews_data.csv').dropna(subset=['comment'])
    #df = merge_skytrax_tripadvisor_data(df_skytrax, df_tripadvisor)

    # Clean column names
    df.columns = df.columns.str.replace('&', 'and')
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.lower()

    # Drop unnamed columns and columns without enough records
    df = df.loc[:, ['unnamed' not in col for col in df.columns]]
    df = df.loc[:, [len(df) - df[col].isna().sum() >= 1000 for col in df.columns]]

    # Merge the disjoint 'aircraft' and 'aircraft_type' columns into one column then clean it
    df.aircraft = (df.aircraft.fillna('') + df.aircraft_type.fillna(''))
    df.aircraft = df.aircraft.replace('', np.nan)
    df.aircraft = df.aircraft.apply(clean_aircraft_string)
    df = df.drop('aircraft_type', axis=1)

    # Clean the header string
    df.header = df.header.apply(clean_header_string)

    # Clean the comment string
    df.comment = df.comment.apply(lambda x: x.lower())
    df.comment = df.apply(lambda x: remove_airline_name_from_comment(x.comment, x.airline), axis=1)
    df.comment = df.apply(lambda x: append_header_to_comment(x.comment, x.header), axis=1)
    df.comment = df.comment.apply(clean_comment_string)

    # Save cleaned dataset to csv file
    df.to_csv('cleaned_reviews.csv', index=None)

    # Create sentence dataframe for topic modeling. Keep_nouns_only=True is time consuming but discards noisy adjectives
    filter_nouns_in_sentences = True
    df['comment_id'] = df.index
    df_sentences = create_sentence_dataframe(df, filter_nouns=filter_nouns_in_sentences)
    df_sentences.to_csv('noun_sentences_for_topic_modeling.csv' if filter_nouns_in_sentences
                        else 'sentences_for_topic_modeling', index=None)

