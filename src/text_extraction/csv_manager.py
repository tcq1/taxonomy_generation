import csv


def export_docs(documents, output_path):
    """ Exports a list of dictionaries to a single csv file

    :param documents: list of dictionaries
    :param output_path: file path to csv file
    """
    for doc in documents:
        export_dict(doc, output_path)


def export_dict(dictionary, output_path):
    """ Exports an ordered dictionary to a csv file

    :param dictionary: dictionary with words as keys and the number of appearances as values
    :param output_path: file path to csv file
    """

    with open(output_path, 'a', encoding='utf-8') as f:
        for key in dictionary.keys():
            try:
                f.write('{}:{},'.format(key, dictionary[key]))
            except UnicodeEncodeError:
                print("Couldn't encode {}. Skip".format(key))
        f.write('\n')

    f.close()


def import_dict(file_path):
    """ Imports a dictionary from a csv file
    :param file_path: file path to csv file
    :return: dictionary
    """

    dictionary = {}

    try:
        with open(file_path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                dictionary[row[0]] = int(row[1])
        f.close()
    except IOError:
        print("Couldn't open file")

    return dictionary
