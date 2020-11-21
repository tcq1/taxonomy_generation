import pyphen
import os
from src.text_extraction.read_pdf import pdf_to_string
import datetime
import pandas as pd
import spacy


def get_number_syllables(word):
    """ Returns number of syllables in the word

    :return: int
    """
    dic = pyphen.Pyphen(lang='de_DE')
    split_word = dic.inserted(word).split('-')

    # if '-' appears in word then there will be empty strings in split_word --> remove empty strings
    while '' in split_word:
        split_word.remove('')

    return len(split_word)


def get_word_length(word):
    """ Returns number of letters in the word

    :return: int
    """
    return len(word)


def has_capital_letter(word):
    """ Returns whether the word starts with a capital letter or not

    :return: boolean
    """
    return word[0].isupper()


def contains_hyphen(word):
    """ Returns whether the word contains a hyphen or not

    :return: boolean
    """
    return '-' in word


def number_appearances_in_texts(word, dict_path):
    """ Returns the number of texts in which the word appears

    :return: int
    """

    counter = 0
    start = datetime.datetime.now()

    df = pd.read_csv(dict_path, encoding='utf-8')
    for index, row in df.iterrows():
        if word in row:
            counter += 1
            break

    end = datetime.datetime.now()
    print('Done after {}'.format(end - start))

    return counter


def appearance_per_doc_length(word, dict_path, nlp):
    """ Returns average amount of appearances compared to the document length (skips documents without appearance)

    :return: float
    """

    avg = 0
    start = datetime.datetime.now()

    df = pd.read_csv(dict_path, encoding='utf-8')

    for index, row in df.iterrows():
        # number of words per document
        doc_length = 0
        counter = 0
        for element in row:
            dic = element.split(':')
            doc_length += dic[1]

            lemma = nlp(word)[0].lemma_

            if lemma == dic[0]:
                counter += dic[1]

        avg += counter / doc_length

    end = datetime.datetime.now()
    print('Done after {}'.format(end-start))

    return avg / len(df.index)
