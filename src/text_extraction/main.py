import pdfminer
import datetime
import os
import spacy

from src.text_extraction.csv_manager import *
from src.text_extraction.read_pdf import *


def add_element_to_dict(dictionary, element):
    """ Adds an element to a dictionary. If not in dictionary, adds a new key to dictionary.
    :param dictionary: dictionary
    :param element: string
    :return: updated dictionary
    """

    if len(element) == 1:
        return dictionary

    if element not in dictionary.keys():
        dictionary[element] = 1
    else:
        dictionary[element] += 1

    return dictionary


def adapt_lengths_of_docs(documents):
    max_length = max(len(doc) for doc in documents)

    for doc in documents:
        empty_key = ' '
        while len(doc) < max_length:
            doc[empty_key] = 0
            empty_key += ' '

    return documents


def main():
    directory_of_files = '../../resources/pdfs'
    documents = []
    documents_lemmas = []
    missing = {}
    start_time = datetime.datetime.now()
    output_csv = '../../output/dictionary_new.csv'
    output_lemmas = '../../output/dictionary_lemmas_new.csv'
    nlp = spacy.load('de_core_news_lg')

    for root, directories, filenames in os.walk(directory_of_files, topdown=False):
        for filename in filenames:
            if filename.lower().endswith('.pdf'):
                word_dict = {}
                lemma_dict = {}
                file_path = os.path.join(root, filename).replace("\\", "/")
                try:
                    doc = nlp(replace_cid_codes(pdf_to_string(file_path)))
                    for token in doc:
                        if token.is_alpha:
                            add_element_to_dict(word_dict, token.text)
                            add_element_to_dict(lemma_dict, token.lemma_)
                    print('File {} processed!'.format(file_path))
                except FileNotFoundError:
                    print('File {} not found. Skip'.format(file_path))
                    missing[file_path] = 'not found'
                except pdfminer.pdfdocument.PDFTextExtractionNotAllowed:
                    print('File {} couldn\'t be converted. Skip'.format(file_path))
                    missing[file_path] = 'not convertible'
                word_dict = {k: v for k, v in sorted(word_dict.items(), key=lambda item: item[1], reverse=True)}
                lemma_dict = {k: v for k, v in sorted(lemma_dict.items(), key=lambda item: item[1], reverse=True)}
                if len(word_dict) > 0:
                    documents.append(word_dict)
                if len(lemma_dict) > 0:
                    documents_lemmas.append(lemma_dict)

    documents = adapt_lengths_of_docs(documents)
    documents_lemmas = adapt_lengths_of_docs(documents_lemmas)

    export_docs(documents, output_csv)
    export_docs(documents_lemmas, output_lemmas)

    end_time = datetime.datetime.now()

    if len(missing.keys()) > 0:
        print('Couldn\'t convert {} pdf files.'.format(len(missing.keys())))
    print('Done after {}'.format(end_time - start_time))
    

if __name__ == '__main__':
    main()
