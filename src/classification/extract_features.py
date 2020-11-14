import pyphen
import os
import pdfminer3
import pdfminer3.pdfdocument
from src.text_extraction.read_pdf import pdf_to_string
import datetime


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


def number_appearances_in_texts(word):
    """ Returns the number of texts in which the word appears

    :return: int
    """
    counter = 0
    directory_of_files = 'resources/pdfs'
    start = datetime.datetime.now()

    for root, directories, filenames in os.walk(directory_of_files, topdown=False):
        for filename in filenames:
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(root, filename).replace("\\", "/")
                try:
                    text = pdf_to_string(file_path)
                    print('{}: {}'.format(file_path, word in text))
                    if word in text:
                        counter += 1
                except FileNotFoundError:
                    print('File {} not found. Skip'.format(file_path))
                except pdfminer3.pdfdocument.PDFTextExtractionNotAllowed:
                    print('File {} couldn\'t be converted. Skip'.format(file_path))

    end = datetime.datetime.now()
    print('Done after {}'.format(end - start))

    return counter


def appearance_per_doc_length(word):
    """ Returns average amount of appearances compared to the document length (skips documents without appearance)

    :return: float
    """

    avg = 0
    counter = 0
    directory_of_files = 'resources/pdfs'
    start = datetime.datetime.now()

    for root, directories, filenames in os.walk(directory_of_files, topdown=False):
        for filename in filenames:
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(root, filename).replace("\\", "/")
                try:
                    text = pdf_to_string(file_path)
                    count = text.count(word)
                    if count > 0:
                        print('{}: {}'.format(file_path, count / len(text)))
                        avg += count / len(text)
                        counter += 1
                    else:
                        print('No appearance of "{}" in {}'.format(word, file_path))
                except FileNotFoundError:
                    print('File {} not found. Skip'.format(file_path))
                except pdfminer3.pdfdocument.PDFTextExtractionNotAllowed:
                    print('File {} couldn\'t be converted. Skip'.format(file_path))

    end = datetime.datetime.now()
    print('Done after {}'.format(end-start))

    return avg / counter
