import pyphen


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
    :param documents: csv file with dictionaries of words from documents
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
