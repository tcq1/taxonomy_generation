import pyphen
import numpy as np


def get_number_syllables(word):
    """ Returns number of syllables in the word

    :param word: word
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

    :param word: word
    :return: int
    """
    return len(word)


def has_capital_letter(word):
    """ Returns whether the word starts with a capital letter or not

    :param word: word
    :return: int(boolean)
    """
    return int(word[0].isupper())


def appearance_per_doc_length(word, documents):
    """ Returns average amount of appearances compared to the document length (skips documents without appearance)
    and number of documents in which the word appears.

    :param word: word
    :param documents: list of dictionaries of words from documents
    :return: [appearance ratio, appearances]
    """

    # initialize appearance ratio
    avg = 0
    # initialize number of appearances
    number_appearances = 0

    for document in documents:
        # number of words in the document
        doc_length = 0
        # number of appearances of the word
        counter = 0

        # get length of doc
        for key, value in document.items():
            doc_length += value

        # count appearances of word
        if word in document.keys():
            number_appearances += 1
            counter += document[word]

        # calculate ratio
        avg += counter / doc_length

    return [avg / len(documents), number_appearances / len(documents)]


def count_word_in_documents(word, documents):
    """ Count number of documents in which the word appears.

    :param word: word
    :param documents: list of dictionaries of words from documents
    :return: int
    """
    counter = 0
    for document in documents:
        if word in document.keys():
            counter += 1

    return counter


def tf_idf(word, documents):
    """ Calculate the term frequency, inverse document frequency score vector.

    :param word: word
    :param documents: list of dictionaries of words from documents
    :return: list of floats
    """
    vector = np.array([])
    for document in documents:
        vector = np.append(document[word])
    vector *= np.log(len(documents) / count_word_in_documents(word, documents))

    return vector


def normed_word_vector(word, nlp):
    """ Returns the L2 norm of the words vector

    :param word: word
    :param nlp: nlp model
    :return: float
    """
    return nlp(word).vector_norm


def get_suffix(word, nlp):
    """ Return suffix of a word.

    :param word: word
    :param nlp: spacy model
    :return: int
    """
    return nlp(word)[0].suffix


def get_prefix(word, nlp):
    """ Return prefix of a word

    :param word: word
    :param nlp: spacy model
    :return: int
    """
    return nlp(word)[0].prefix


def is_stop_word(word, nlp):
    """ Check if a word is a stop word.

    :param word: word
    :param nlp: spacy model
    :return: boolean
    """
    return nlp(word)[0].is_stop
