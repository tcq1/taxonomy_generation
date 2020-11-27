from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from timeit import default_timer as timer

from src.classification.extract_features import *
from src.text_extraction.csv_manager import import_docs


documents = None
wikipedia = None


def get_feature_vector(word):
    appearance_ratio_pdfs, appearances_pdfs = appearance_per_doc_length(word, documents)
    appearance_ratio_wikipedia, appearance_wikipedia = appearance_per_doc_length(word, wikipedia)
    return [has_capital_letter(word), get_word_length(word), get_number_syllables(word),
            appearance_ratio_pdfs, appearances_pdfs,
            appearance_ratio_wikipedia, appearance_wikipedia]


def get_feature_vector_of_list(words):
    features = []

    for word in words:
        features.append(get_feature_vector(word))

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
    feature_names = ['Has capital letter', 'Word length', 'Number syllables',
                     'Appearance ratio in pdfs', 'Appearances in pdfs',
                     'Appearance ratio in wikipedia articles', 'Appearances in wikipedia articles']
    importance = clf.feature_importances_
    importance_dict = {}
    for i in range(len(feature_names)):
        importance_dict[feature_names[i]] = importance[i] * 100

    return importance_dict


def get_data(positive_path, negative_path, test_size):
    """

    :param positive_path: File path of positive labeled data
    :param negative_path: File path of negative labeled data
    :param test_size: Value between 0 and 1 determining the proportion of validation data
    :return:
    """
    positive_data, labels = load_data(positive_path, 1)
    negative_data, labels2 = load_data(negative_path, 0)

    training_data = positive_data
    training_data.extend(negative_data)
    labels.extend(labels2)

    return train_test_split(extract_feature_vector_of_list(training_data), labels,
                            test_size=test_size, random_state=1)


def train(X, y, clf, output_path):
    """ Trains a model and returns it
    :param X: Input data array of shape [n_samples, n_features]
    :param y: Label array of shape [n_samples]
    :param clf: Classifier
    :param output_path: Output path for model
    :return: model
    """

    start_time = timer()
    score = cross_val_score(clf, extract_feature_vector_of_list(X), y, cv=10, scoring='f1_macro')
    print('Score = {}, mean = {}'.format(score, score.mean()))
    print('Importance: {}'.format(get_feature_importance(clf)))
    end_time = timer()
    print('Total time for training: {}s'.format(end_time-start_time))

    file_path = output_path
    export_model(clf, file_path)

    return clf


def predict(model, word):
    """ Makes prediction for a single word

    :param model: model
    :param word: word
    :return: 0 for negative, 1 for positive
    """

    return model.predict_proba([get_feature_vector(word)])


def main():
    global documents
    global wikipedia
    positive_path = '../../output/training_small_positive.txt'
    negative_path = '../../output/training_small_negative.txt'
    documents_path = '../../output/dictionary_lemmas.csv'
    wikipedia_path = '../../output/wikipedia_lemmas.csv'
    documents = import_docs(documents_path)
    wikipedia = import_docs(wikipedia_path)

    clf = RandomForestClassifier(n_estimators=1000, max_features=4, max_depth=6, min_samples_split=2, n_jobs=8)
    X_training, X_validation, y_training, y_validation = get_data(positive_path, negative_path, 0.3)

    model_path = '../../output/clf.joblib'
    model = train(X_training, y_training, clf, model_path)


if __name__ == '__main__':
    main()
