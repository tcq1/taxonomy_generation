import spacy

from src.text_extraction.csv_manager import import_docs
from shutil import copyfile


def string_in_file(file_path, string):
    """ Checks if string is already in the file

    :param file_path: file path of file that has to be checked
    :param string: string to look for
    :return: boolean
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if string in line:
                return True

    return False


def extend_training_data(source_path, original_path, output_path):
    """ Makes a copy of the original training data and appends extra data to the copy
    :param source_path: csv file with words that should be appended to the file
    :param original_path: path of original file that should be extended
    :param output_path: path for the new file
    """

    dictionaries = import_docs(source_path)
    copyfile(original_path, output_path)

    with open(output_path, 'a', encoding='utf-8') as f:
        for dictionary in dictionaries:
            for key in dictionary.keys():
                if not string_in_file(output_path, key):
                    try:
                        f.write('{}\n'.format(key))
                    except UnicodeEncodeError:
                        print("Couldn't encode {}. Skip".format(key))


def transform_to_lemma(original_path, output_path):
    """ Makes a copy of the original training data and makes a new file with the lemmas of all the original words.

    :param original_path: path of original file
    :param output_path: path for the new file
    """
    file_old = open(original_path, 'r', encoding='utf-8')
    file_new = open(output_path, 'w', encoding='utf-8')

    print('Loading nlp model...')
    nlp = spacy.load('de_core_news_lg')
    print('nlp model loaded!')

    print('Writing to new file...')
    for line in file_old:
        file_new.write('{}\n'.format(nlp(line)[0].lemma_))
    print('Done!')

    file_old.close()
    file_new.close()


def load_words_to_list(file_path):
    """ Reads a file line by line and adds words to a list

    :param file_path: Path to file
    :return: list
    """
    
    word_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word_list.append(line.split('\n')[0])

    f.close()
    
    return word_list


def add_label_to_words(words, label, file_path):
    """ Takes a list of words, adds labels to them and exports them to a csv file
    Assumption: All words have the same label

    :param words: list of words
    :param label: label
    :param file_path: output file path
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for word in words:
            f.write('{},{}\n'.format(word, label))

    f.close()


def main():
    # csv_file = '../../output/csv/dictionary_lemmas.csv'
    # original = '../../output/training_data/empty'
    # output_path = '../../output/training_data/predicted_data.txt'
    # extend_training_data(csv_file, original, output_path)

    # old = '../../output/training_data/training_small_positive.txt'
    # new = '../../output/training_data/training_small_positive_lemmas.txt'
    # transform_to_lemma(old, new)

    # print(load_words_to_list('../../output/training_data/training_small_positive.txt'))

    input_path = '../../output/training_data/training_small_negative_extended2.txt'
    output_path = '../../output/training_data/training_large_negative_labeled.txt'
    add_label_to_words(load_words_to_list(input_path), 0, output_path)


if __name__ == '__main__':
    main()
