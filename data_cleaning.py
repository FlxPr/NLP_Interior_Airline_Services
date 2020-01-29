import pandas as pd
import numpy as np
import re
import nltk
from spellchecker import SpellChecker
import gensim
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import numpy as np
from gensim import corpora, models
from pprint import pprint
import itertools


# nltk.download('wordnet')
# nltk.download('punk')

def clean_aircraft_string(aircraft_string):
    if type(aircraft_string) == str:
        aircraft_string = aircraft_string.lower()
        aircraft_string = aircraft_string.replace('airbus', '')
        aircraft_string = aircraft_string.replace('and', '/')
        aircraft_string = aircraft_string.replace('embraer ', 'e')
        aircraft_string = aircraft_string.strip()
        return aircraft_string
    return np.nan


def clean_header_string(header_string):
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
    if '|' in comment_string:
        if 'verified' in comment_string.split('|')[0].lower():
            return comment_string.split('|')[1]
    return comment_string


def reduce_word_exaggeration(comment_string):
    exaggeration = re.compile(r"(.)\1{2,}")
    return exaggeration.sub(r"\1\1", comment_string)


def remove_characters(comment_string):
    comment_string = re.sub('\s+', ' ', comment_string)
    comment_string = re.sub("\'", "", comment_string)
    return comment_string


def remove_airline_name_from_comment(comment, airline_name):
    airline_name_words = airline_name.lower().split(' ')
    for number_of_words in range(1, len(airline_name_words) + 1)[::-1]:
        comment = comment.replace(' '.join(airline_name_words[:number_of_words]), '')
    return comment.replace('boeing', '').replace('airbus', '').replace('airline', '').replace('leg room', 'legroom')


def append_header_to_comment(comment, header, twice=True):
    if type(header) == str:
        return '{}. {}. {}'.format(header, header, comment) if twice else '{}. {}'.format(header, comment)
    return comment


def clean_comment_string(comment_string):
    comment_string = delete_verified_review_prefix(comment_string)
    comment_string = reduce_word_exaggeration(comment_string)
    comment_string = remove_characters(comment_string)
    # TODO put other functions in there
    return comment_string


def tokenize_lemma_stem(comment):
    result = []
    for token in gensim.utils.simple_preprocess(comment):
        if token not in gensim.parsing.preprocessing.STOPWORDS\
                and (len(token) > 3
                     or token in ['leg', 'arm', 'eye', 'old', 'bag', 'hip', 'eat',
                                  'row', 'low', 'big', 'age', 'hot', 'kid', 'gap']):
            result.append(PorterStemmer().stem(WordNetLemmatizer().lemmatize(token, pos='n')))
    return result


def preprocess_for_topic_modeling(data):
    data = data.copy()
    word_document_sets = [set([word for word in comment.split(' ')]) for comment in data.comment]


# def spell_correct(data_frame):  # TODO integrate in tokenizer
#     data_frame = data_frame.copy()
#     spell = SpellChecker()
#     spell.word_frequency.load_words(data_frame.aircraft.dropna().unique())
#     spell.known(['a380'])
#     words = set.union(*[set(comment.split(' ')) for comment in df.comment.str.lower()])


def create_sentence_dataframe(comment_dataframe):
    comment_dataframe.comment = comment_dataframe.comment.apply(delete_verified_review_prefix)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    id_sentence_list = []
    for _, comment_row in comment_dataframe.iterrows():
        sentences = tokenizer.tokenize(comment_row['comment'])
        id_sentence_list.append(zip([comment_row['comment_id']] * len(sentences), sentences))
    id_sentence_list = itertools.chain.from_iterable(id_sentence_list)
    id_sentence_dataframe = pd.DataFrame(id_sentence_list, columns=['comment_id', 'sentence'])
    return id_sentence_dataframe


if __name__ == '__main__':
    df = pd.read_csv('skytrax_reviews_data.csv')
    df['comment_id'] = df.index

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

    df.header = df.header.apply(clean_header_string)

    df.comment = df.comment.apply(lambda x: x.lower())
    df.comment = df.comment.apply(clean_comment_string)
    df.comment = df.apply(lambda x: append_header_to_comment(x.comment, x.header), axis=1)
    df.comment = df.apply(lambda x: remove_airline_name_from_comment(x.comment, x.airline), axis=1)

    df_sentences = create_sentence_dataframe(df)

    df_sentences['comment_token_stem_lemma'] = df_sentences.sentence.apply(tokenize_lemma_stem)
    df.to_csv('cleaned_reviews.csv', index=None)
    df_sentences.to_csv('sentences_for_topic_modelling.csv', index=None)
