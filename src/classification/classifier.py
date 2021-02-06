import numpy as np
import spacy

from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from timeit import default_timer as timer

from src.classification.extract_features import *
from src.text_extraction.csv_manager import import_docs
from src.text_extraction.manipulate_training_data import load_words_to_list


documents = None
wikipedia = None
nlp = spacy.load('de_core_news_lg')


def get_feature_vector(word):
    """ Get feature vector of a word.

    :param word: string
    :return: list of floats
    """
    appearance_ratio_pdfs, appearances_pdfs = appearance_per_doc_length(word, documents)
    appearance_ratio_wikipedia, appearance_wikipedia = appearance_per_doc_length(word, wikipedia)
    return np.array([has_capital_letter(word),
                     get_word_length(word),
                     get_number_syllables(word),
                     appearance_ratio_pdfs, appearances_pdfs,
                     appearance_ratio_wikipedia, appearance_wikipedia,
                     normed_word_vector(word, nlp),
                     get_suffix(word, nlp)])


def get_feature_names():
    """ Return a list of the features as strings.

    :return: list of strings
    """
    return ['Has capital letter',
            'Word length',
            'Number syllables',
            'Appearance ratio in pdfs', 'Appearances in pdfs',
            'Appearance ratio in wikipedia articles', 'Appearances in wikipedia articles',
            'Feature vector normed',
            'Suffix']


def get_feature_vector_of_list(words):
    """ Get the feature vectors of all words in a list.

    :param words: list of words
    :return: list of feature vectors
    """
    return np.array([get_feature_vector(word) for word in words])


def load_data(path):
    """ Load input data from a given path. Data is stored as word,label\nword,label\n ...

    :param path: path to file
    :return: (list of words, list of labels)
    """
    words = []
    labels = []
    with open(path, encoding='utf-8') as f:
        lines = f.read().splitlines()
        for line in lines:
            l = line.split(',')
            words.append(l[0])
            labels.append((l[1]))

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
    feature_names = get_feature_names()
    importance = clf.feature_importances_
    importance_dict = {}
    for i in range(len(feature_names)):
        importance_dict[feature_names[i]] = importance[i] * 100

    return importance_dict


def normalize_data(x_train, x_validate):
    """ Min-max normalize the datasets.

    :param x_train: List of feature vectors of training data
    :param x_validate: List of feature vectors of validation data
    :return: x_train, x_validate
    """
    min_x = x_train.min(axis=0)
    max_x = x_train.max(axis=0)
    x_train = (x_train - min_x) / (max_x - min_x)
    x_validate = (x_validate - min_x) / (max_x - min_x)

    return x_train, x_validate


def get_data(file_paths, test_size):
    """ Load training data and split up in test and validation data sets.

    :param file_paths: List of paths to iterate through
    :param test_size: Value between 0 and 1 determining the proportion of validation data
    :return: (x_train, y_train), (x_validate, y_validate)
    """

    training_data = []
    labels = []

    for file in file_paths:
        data, label = load_data(file)
        training_data.extend(data)
        labels.extend(label)

    x_train, x_validate, y_train, y_validate = train_test_split(get_feature_vector_of_list(training_data), labels,
                                                                test_size=test_size, random_state=1)

    x_train, x_validate = normalize_data(x_train, x_validate)

    return x_train, x_validate, y_train, y_validate


def print_feature_importance(clf):
    """ Prints out feature importance of a model

    :param clf: Classifier
    """
    importance = get_feature_importance(clf)
    print('Importance: ')
    for key in importance.keys():
        print('{}: {} %'.format(key, importance[key]))


def train(X, y, clf, k, output_path, scoring):
    """ Trains a model and returns it
    :param X: Input data array of shape [n_samples, n_features]
    :param y: Label array of shape [n_samples]
    :param clf: Classifier
    :param k: Number of subsets to divide the training data into for cross validation
    :param output_path: Output path for model
    :param scoring: scoring measure for k-fold cv
    :return: model
    """
    clf.fit(X, y)
    score = cross_val_score(clf, X, y, cv=k, scoring=scoring)
    print('Score = {}, mean = {}'.format(score, score.mean()))
    try:
        print_feature_importance(clf)
    except AttributeError:
        print("Couldn't get feature importance for this classifier!")

    file_path = output_path
    export_model(clf, file_path)

    return clf


def find_optimal_classifier(clf, X_train, y_train, param_grid, k, scoring):
    """ Uses randomized search to find the best classifier.

    :param clf: Model
    :param X_train: Training data input vectors
    :param y_train: Training data labels
    :param param_grid: Parameter grid in which to search for
    :param k: Number of subsets to divide the training data into for cross validation
    :param scoring: Scoring method
    :return: Best model
    """

    print('Starting to find optimal classifier...')

    start = timer()
    search = GridSearchCV(clf, param_grid=param_grid, cv=k, scoring=scoring, n_jobs=8)
    search.fit(X_train, y_train)
    end = timer()

    print(search.cv_results_)
    print('Optimal parameters: '.format(search.get_params()))
    print('Best score: '.format(search.best_score_))
    print('Tuning hyperparameters took {}s'.format(end-start))

    return search.best_estimator_


def predict_words(words, clf):
    """ Makes prediction for a list of words and returns a list of positive labeled words.

    :param words: list of words
    :param clf: classifier
    :return: list of words
    """
    if wikipedia is None or documents is None:
        initialize_datasets()

    predictions = clf.predict(get_feature_vector_of_list(words))

    output_words = []
    for i in range(len(words)):
        if predictions[i] == '1':
            output_words.append(words[i])

    return output_words


def export_predicted_words(words, clf, file_path):
    """ Makes prediction for a list of words and exports them to a csv file

    :param words: list of words
    :param clf: Classifier
    :param file_path: export file path
    """
    print('Making predictions for words...')
    y_new = clf.predict(get_feature_vector_of_list(words))

    print('Write to file...')
    with open(file_path, 'w', encoding='utf-8') as f:
        for i in range(len(words)):
            f.write('{},{}\n'.format(words[i], y_new[i]))
    f.close()


def initialize_datasets():
    """ Initialize the datasets that are necessary for getting the feature vectors.
    """
    documents_path = '../../output/csv/dictionary_lemmas.csv'
    wikipedia_path = '../../output/csv/wikipedia_lemmas.csv'

    global documents
    global wikipedia
    documents = import_docs(documents_path)
    wikipedia = import_docs(wikipedia_path)


def load_datasets():
    """ Load datasets, split them in training and validation sets and normalize the features.

    :return: X_training, X_validation, y_training, y_validation
    """
    positive_path = '../../output/training_data/training_small_positive_lemmas_labeled.txt'
    negative_small = '../../output/training_data/training_small_negative_labeled.txt'
    negative_medium = '../../output/training_data/training_medium_negative_labeled.txt'
    negative_large = '../../output/training_data/training_large_negative_labeled.txt'
    extra_data = '../../output/training_data/done.csv'
    initialize_datasets()

    # get split datasets
    return get_data([positive_path, negative_large, extra_data], 0.3)


def main():
    """ Load training data
    """
    start_time = timer()
    x_train, x_validate, y_train, y_validate = load_datasets()
    print(f"Done loading datasets after {timer() - start_time}s.")

    """ Model
    """
    # instantiate classifier
    clf = RandomForestClassifier(n_estimators=1000, max_features=4, max_depth=6, min_samples_split=2, n_jobs=8,
                                 class_weight='balanced')

    # clf = MLPClassifier(max_iter=1000, tol=1e-4)

    # set some parameters
    scoring = 'f1_macro'
    k = 6
    model_path = '../../output/models/random_forest.joblib'

    # parameter_grid = {'max_features': [3, 4, 5, 6],
    #                   'max_depth': [5, 6, 7, 8],
    #                   'min_samples_split': [2, 3, 4]}

    # model = find_optimal_classifier(clf, X_training, y_training, parameter_grid, k, scoring)

    # train
    model = train(x_train, y_train, clf, k, model_path, scoring)

    # evaluation
    y_predicted = model.predict(x_validate)
    cm = confusion_matrix(y_validate, y_predicted)
    print('[[{}  {}]   --> {}\n [{}  {}]]   --> {}'.format(cm[0][0], cm[0][1], cm[0][0] / sum(cm[0]),
                                                           cm[1][0], cm[1][1], cm[1][1] / sum(cm[1])))
    end_time = timer()
    print('Total time for training: {}'.format(end_time - start_time))

    """ Load model and predict
    """
    # start = timer()
    # model = load(model_path)
    # X_new = load_words_to_list('../../output/training_data/predicted_data.txt')
    # export_predicted_words(X_new, model, '../../output/training_data/predicted_and_label.csv')
    # end = timer()
    # print('Prediction and export took {}s!'.format(end-start))


if __name__ == '__main__':
    main()
