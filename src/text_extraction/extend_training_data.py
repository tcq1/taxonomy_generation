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


def main():
    """ Makes a copy of the original training data and appends extra data to the copy
    """
    csv_file = '../../output/wikipedia_lemmas.csv'
    dictionaries = import_docs(csv_file)

    original = '../../output/training_small_negative_extended.txt'
    output_path = '../../output/training_small_negative_extended2.txt'

    copyfile(original, output_path)

    with open(output_path, 'a', encoding='utf-8') as f:
        for dictionary in dictionaries:
            for key in dictionary.keys():
                if not string_in_file(output_path, key):
                    try:
                        f.write('{}\n'.format(key))
                    except UnicodeEncodeError:
                        print("Couldn't encode {}. Skip".format(key))


if __name__ == '__main__':
    main()
