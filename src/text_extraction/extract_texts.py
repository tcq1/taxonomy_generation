import os
import pdfminer
import spacy
import wikipedia

from newsapi import NewsApiClient
from timeit import default_timer as timer

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


def get_tokens(nlp, text):
    """ Extracts tokens from a text and returns a dictionary with the tokens and the number of appearances and
    a dictionary with the lemmas of the tokens.

    :param nlp: spacy model
    :param text: Text to extract the tokens from
    :return: word_dict, lemma_dict
    """
    word_dict = {}
    lemma_dict = {}

    doc = nlp(text)
    for token in doc:
        if token.is_alpha:
            add_element_to_dict(word_dict, token.text)
            add_element_to_dict(lemma_dict, token.lemma_)

    return word_dict, lemma_dict


def extract_pdfs(nlp):
    """ Extracts all tokens from pdf files and exports them to csv files. One csv file contains the words unmodified,
    the other one contains the lemmas of the words.

    :param nlp: spacy model
    """

    # input and output paths
    directory_of_files = '../../resources/pdfs'
    output_csv = '../../output/dictionary.csv'
    output_lemmas = '../../output/dictionary_lemmas.csv'

    # setup
    documents = []
    documents_lemmas = []
    missing = {}

    start_time = timer()

    # iterate over all files in all subfolders of given input path
    for root, directories, filenames in os.walk(directory_of_files, topdown=False):
        for filename in filenames:
            # only process pdf files
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(root, filename).replace("\\", "/")
                # try to extract words from pdfs
                try:
                    word_dict, lemma_dict = get_tokens(nlp, replace_cid_codes(pdf_to_string(file_path)))
                    # sort dictionaries descending by appearance of tokens
                    word_dict = {k: v for k, v in sorted(word_dict.items(), key=lambda item: item[1], reverse=True)}
                    lemma_dict = {k: v for k, v in sorted(lemma_dict.items(), key=lambda item: item[1], reverse=True)}
                    # add dicts to lists
                    documents.append(word_dict)
                    documents_lemmas.append(lemma_dict)
                    print('File {} processed!'.format(file_path))
                except FileNotFoundError:
                    print('File {} not found. Skip'.format(file_path))
                    missing[file_path] = 'not found'
                except pdfminer.pdfdocument.PDFTextExtractionNotAllowed:
                    print('File {} couldn\'t be converted. Skip'.format(file_path))
                    missing[file_path] = 'not convertible'

    # export lists to csv files
    export_docs(documents, output_csv)
    export_docs(documents_lemmas, output_lemmas)

    end_time = timer()

    if len(missing.keys()) > 0:
        print('Couldn\'t convert {} pdf files.'.format(len(missing.keys())))
    print('Done after {}s'.format(end_time - start_time))
    

def extract_wikipedia(nlp, output_csv, output_lemmas, num_pages):
    """ Extracts all tokens from random wikipedia pages and exports them to csv files.
        One csv file contains the words unmodified, the other one contains the lemmas of the words.

        :param nlp: spacy model
        :param output_csv: output path for normal token.text
        :param output_lemmas: output path for token.lemma_
        :param num_pages: number of wikipedia pages to scan
    """

    # setup
    wikipedia.set_lang('de')
    wikipedia_articles = []
    wikipedia_articles_lemmas = []

    start_time = timer()

    # iterate num_pages times
    i = 0
    while i < num_pages:
        i += 1
        # get random wikipedia page
        try:
            page = wikipedia.page(title=wikipedia.random())
        except wikipedia.DisambiguationError:
            # skip if DisambiguationError appears and decrement counter
            i -= 1
            print('DisambiguationError! Skip...')
            continue
        except wikipedia.PageError:
            # skip if PageError appears and decrement counter
            i -= 1
            print('PageError! Skip...')
            continue

        # extract words and lemmas from page content
        word_dict, lemma_dict = get_tokens(nlp, page.content)
        # sort dictionaries descending by appearance of tokens
        word_dict = {k: v for k, v in sorted(word_dict.items(), key=lambda item: item[1], reverse=True)}
        lemma_dict = {k: v for k, v in sorted(lemma_dict.items(), key=lambda item: item[1], reverse=True)}
        # add tokens of pages to list
        if len(word_dict) > 0:
            wikipedia_articles.append(word_dict)
        if len(lemma_dict) > 0:
            wikipedia_articles_lemmas.append(lemma_dict)
        print('[{}/{}]: Page {} processed!'.format(i, num_pages, page.title))

    # export lists to csv files
    export_docs(wikipedia_articles, output_csv)
    export_docs(wikipedia_articles_lemmas, output_lemmas)

    end_time = timer()
    print('Done after {}s'.format(end_time - start_time))


def extract_news_articles(nlp, output_csv, output_lemmas):
    """ Extracts all tokens from title and descriptions of German news article providers.

    :param nlp: spacy model
    :param output_csv: output path for normal token.text
    :param output_lemmas: output path for token.lemma_
    """
    # initialize newsapi
    api_key = '2f4a6aa461194cce948ded99f35fae6f'
    newsapi = NewsApiClient(api_key)

    start_time = timer()

    # get German news providers
    sources = [src['id'] for src in newsapi.get_sources(language='de')['sources']]
    sources_string = sources[0]
    for i in range(1, len(sources)):
        sources_string += ',{}'.format(sources[i])

    # get articles
    articles = newsapi.get_everything(language='de', sources=sources_string, page_size=100)
    print('Total results: {}'.format(articles['totalResults']))

    news_articles = []
    news_articles_lemmas = []

    # TODO: add param to specify number of articles

    for article in articles['articles']:
        # get article content
        content = article['title'] + '. ' + article['description']
        if article['content'] is not None:
            content += '. ' + article['content']

        # extract words and lemmas from article
        word_dict, lemma_dict = get_tokens(nlp, content)
        # sort dictionaries descending by appearance of tokens
        word_dict = {k: v for k, v in sorted(word_dict.items(), key=lambda item: item[1], reverse=True)}
        lemma_dict = {k: v for k, v in sorted(lemma_dict.items(), key=lambda item: item[1], reverse=True)}
        # add tokens of pages to list
        if len(word_dict) > 0:
            news_articles.append(word_dict)
        if len(lemma_dict) > 0:
            news_articles_lemmas.append(lemma_dict)

    # export lists to csv files
    export_docs(news_articles, output_csv)
    export_docs(news_articles_lemmas, output_lemmas)

    end_time = timer()
    print('Done after {}s'.format(end_time - start_time))


def main():
    # load spacy model
    nlp = spacy.load('de_core_news_lg')

    # output paths
    output_csv = '../../output/csv/news.csv'
    output_lemmas = '../../output/csv/news_lemmas.csv'

    # extract_wikipedia(output_csv, output_lemmas, 1000)
    extract_news_articles(nlp, output_csv, output_lemmas)


if __name__ == '__main__':
    main()
