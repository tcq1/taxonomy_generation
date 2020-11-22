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

    f.close()


def import_docs(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().splitlines()

    f.close()

    docs = []
    for line in content:
        docs.append(import_dict(line))

    return docs


def import_dict(line):
    """ Converts line of a csv file to a dict
    :param line: line of csv file
    :return: dictionary
    """

    dictionary = {}

    elements = line.split(',')
    for element in elements:
        key, value = element.split(':')
        dictionary[key] = int(value)

    return dictionary
