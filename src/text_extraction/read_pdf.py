from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import TextConverter

import io
import re


def pdf_to_string(path):
    """ Converts a pdf file to a string. String contains
    :param path: Path to pdf file
    :return: String
    """
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    with open(path, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)

        text = fake_file_handle.getvalue()

    # close open handles
    converter.close()
    fake_file_handle.close()

    return text


def replace_cid_codes(string):
    """ Takes a string and replaces relevant cid codes
    :param string: string with cid codes
    :return: string with relevant cid codes replaced
    """

    # characters
    string = string.replace('(cid:160)', ' ')
    string = string.replace('(cid:150)', '-')
    string = string.replace('(cid:146)', "'")
    string = string.replace('(cid:132)', '"')
    string = string.replace('(cid:147)', '"')

    # letters
    string = string.replace('(cid:228)', 'ä')
    string = string.replace('(cid:246)', 'ö')
    string = string.replace('(cid:252)', 'ü')

    string = string.replace('(cid:214)', 'Ö')
    string = string.replace('(cid:220)', 'Ü')
    string = string.replace('(cid:223)', 'ß')

    string = string.replace('\n', ' ')
    string = string.replace('\r', '')

    return string


def split_string(string):
    """ Returns a list of words, split up at certain characters
    :param string:
    :return: list
    """

    return re.split('; |, |\*|\n|\s+|\.', string)
