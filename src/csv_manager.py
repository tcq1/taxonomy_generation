import csv


def export_dict(dictionary, output_path):
    """ Exports an ordered dictionary to a csv file
    """

    with open(output_path, 'w') as f:
        for key in dictionary.keys():
            try:
                f.write("%s,%s\n" % (key, dictionary[key]))
            except UnicodeEncodeError:
                print("Couldn't encode {}. Skip".format(key))


def import_dict(file_path):
    """ Imports a dictionary from a csv file
    :param file_path: file path to csv file
    :return: dictionary
    """

    dictionary = {}

    try:
        with open(file_path, newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                dictionary[row[0]] = int(row[1])
    except IOError:
        print("Couldn't open file")

    return dictionary
