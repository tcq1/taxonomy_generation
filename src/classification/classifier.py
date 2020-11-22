from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from timeit import default_timer as timer

from src.classification.extract_features import *
from src.text_extraction.csv_manager import import_docs


documents = None
wikipedia = None


def extract_feature_vector(word):
    appearance_ratio_pdfs, appearances_pdfs = appearance_per_doc_length(word, documents)
    appearance_ratio_wikipedia, appearance_wikipedia = appearance_per_doc_length(word, wikipedia)
    return [has_capital_letter(word), contains_hyphen(word), get_word_length(word), get_number_syllables(word),
            appearance_ratio_pdfs, appearances_pdfs,
            appearance_ratio_wikipedia, appearance_wikipedia]


def extract_feature_vector_of_list(words):
    features = []

    for word in words:
        features.append(extract_feature_vector(word))

    return features


def load_data(path, label):
    words = []
    labels = []
    with open(path, encoding='utf-8') as f:
        lines = f.read().splitlines()
        for line in lines:
            words.append(line)
            labels.append(label)

    return words, labels


def export_model(model, file_path):
    """ Exports a model

    :param model: Classifier
    :param file_path: Output filepath
    """

    dump(model, file_path)


def import_model(file_path):
    """ Imports a model and returns it

    :param file_path: Filepath of model
    :return: model
    """

    return load(file_path)


def get_feature_importance(clf):
    """ Returns a dictionary with the feature name as key and the importance for classification in % as value

    :param clf: model
    :return: importance dictionary
    """
    feature_names = ['Has capital letter', 'Contains hyphen', 'Word length', 'Number syllables',
                     'Appearance ratio in pdfs', 'Appearances in pdfs',
                     'Appearance ratio in wikipedia articles', 'Appearances in wikipedia articles']
    importance = clf.feature_importances_
    importance_dict = {}
    for i in range(len(feature_names)):
        importance_dict[feature_names[i]] = importance[i] * 100

    return importance_dict


def train(output_path):
    """ Trains a model and returns it
    :param output_path: Output path for model
    :return: model
    """
    positive_path = '../../output/training_small_positive.txt'
    negative_path = '../../output/training_small_negative.txt'

    positive_data, labels = load_data(positive_path, 1)
    negative_data, labels2 = load_data(negative_path, 0)

    training_data = positive_data
    training_data.extend(negative_data)
    labels.extend(labels2)

    start_time = timer()

    clf = RandomForestClassifier(n_estimators=1000, max_features=4, max_depth=6, min_samples_split=2, n_jobs=8)
    clf.fit(extract_feature_vector_of_list(training_data), labels)
    score = cross_val_score(clf, extract_feature_vector_of_list(training_data), labels, cv=10)
    print('Score = {}, mean = {}'.format(score, score.mean()))
    print('Importance: {}'.format(get_feature_importance(clf)))
    end_time = timer()
    print('Total time for training: {}s'.format(end_time-start_time))

    file_path = output_path
    export_model(clf, file_path)

    return clf


def predict(model_path, word):
    clf = load(model_path)
    result = clf.predict([extract_feature_vector(word)])

    return result


def main():
    global documents
    global wikipedia
    documents_path = '../../output/dictionary_lemmas.csv'
    wikipedia_path = '../../output/wikipedia_lemmas.csv'
    documents = import_docs(documents_path)
    wikipedia = import_docs(wikipedia_path)

    model_path = '../../output/clf.joblib'
    train(model_path)
    prediction = predict(model_path, 'Test')
    print('Prediction of word {}: {}'.format('Test', prediction))
    prediction = predict(model_path, 'schweißen')
    print('Prediction of word {}: {}'.format('schweißen', prediction))


if __name__ == '__main__':
    main()
