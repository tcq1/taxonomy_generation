import pdfminer3
import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from src.csv_manager import *
from src.read_pdf import *


def add_element_to_dict(dictionary, element):
    """ Adds an element to a dictionary. If not in dictionary, adds a new key to dictionary.
    :param dictionary: dictionary
    :param element: string
    :return: updated dictionary
    """

    if element not in dictionary.keys():
        dictionary[element] = 1
    else:
        dictionary[element] += 1

    return dictionary


def main():
    directory_of_files = 'resources/pdfs'
    word_dict = {}
    missing = {}
    start_time = datetime.datetime.now()
    output_csv = 'output/dictionary.csv'

    for root, directories, filenames in os.walk(directory_of_files, topdown=False):
        for filename in filenames:
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(root, filename).replace("\\", "/")
                try:
                    words = split_string(replace_cid_codes(pdf_to_string(file_path)))
                    for word in words:
                        add_element_to_dict(word_dict, word)
                    print('File {} processed!'.format(file_path))
                except FileNotFoundError:
                    print('File {} not found. Skip'.format(file_path))
                    missing[file_path] = 'not found'
                except pdfminer3.pdfdocument.PDFTextExtractionNotAllowed:
                    print('File {} couldn\'t be converted. Skip'.format(file_path))
                    missing[file_path] = 'not convertible'

    word_dict = {k: v for k, v in sorted(word_dict.items(), key=lambda item: item[1], reverse=True)}
    end_time = datetime.datetime.now()

    print(word_dict)
    export_dict(word_dict, output_csv)
    if len(missing.keys()) > 0:
        print('Couldn\'t convert {} pdf files.'.format(len(missing.keys())))
    print('Done after {}'.format(end_time - start_time))


if __name__ == '__main__':
    main()
