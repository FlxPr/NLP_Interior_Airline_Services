import pandas as pd
import numpy as np
import re
from spellchecker import SpellChecker


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


def clean_comment_string(comment_string):
    comment_string = delete_verified_review_prefix(comment_string)
    comment_string = reduce_word_exaggeration(comment_string)
    # TODO put other functions in there
    return comment_string


# def spell_correct(data_frame):  # TODO integrate in tokenizer
#     data_frame = data_frame.copy()
#     spell = SpellChecker()
#     spell.word_frequency.load_words(data_frame.aircraft.dropna().unique())
#     spell.known(['a380'])
#     words = set.union(*[set(comment.split(' ')) for comment in df.comment.str.lower()])


if __name__ == '__main__':
    df = pd.read_csv('skytrax_reviews_data.csv')

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
    df.header = df.header.apply(lambda x: np.nan if 'customer review' in x.lower() else x.lower())

    df.comment = df.comment.apply(lambda x: x.lower())
    df.comment = df.comment.apply(clean_comment_string)



