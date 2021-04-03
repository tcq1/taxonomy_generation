import numpy as np
import spacy

from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV, GridSearchCV
from timeit import default_timer as timer

from src.classification.extract_features import *
from src.text_extraction.csv_manager import import_docs
from src.classification.evaluation import calc_recall, calc_precision, calc_f1


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
                     get_suffix(word, nlp),
                     get_prefix(word, nlp),
                     is_stop_word(word, nlp)
                     ])


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
            'Suffix',
            'Prefix',
            'Is stop word']


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
            labels.append((int(l[1])))

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


def normalize_data(x_train, x_validate, file_path):
    """ Min-max normalize the datasets.

    :param x_train: List of feature vectors of training data
    :param x_validate: List of feature vectors of validation data
    :param file_path: file path to directory where to store min and max values
    :return: x_train, x_validate
    """
    min_x = np.min(x_train, axis=0)
    max_x = np.max(x_train, axis=0)

    np.savetxt(file_path + 'min_x', min_x)
    np.savetxt(file_path + 'max_x', max_x)

    x_train = (x_train - min_x) / (max_x - min_x)
    x_validate = (x_validate - min_x) / (max_x - min_x)

    return x_train, x_validate


def get_data(file_paths, test_size, export_path=None):
    """ Load training data and split up in test and validation data sets.

    :param file_paths: List of paths to iterate through
    :param test_size: Value between 0 and 1 determining the proportion of validation data
    :param export_path: If specified store data in following structure: output_path/x_train.csv, output_path/...
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

    normalize_data(x_train, x_validate, 'datasets/default/')

    if export_path is not None:
        np.savetxt(f'{export_path}/x_train.csv', x_train, delimiter=',')
        np.savetxt(f'{export_path}/x_validate.csv', x_validate, delimiter=',')
        np.savetxt(f'{export_path}/y_train.csv', y_train, delimiter=',')
        np.savetxt(f'{export_path}/y_validate.csv', y_validate, delimiter=',')

    return x_train, x_validate, y_train, y_validate


def print_feature_importance(clf):
    """ Prints out feature importance of a model

    :param clf: Classifier
    """
    importance = get_feature_importance(clf)
    print('Importance: ')
    for key in importance.keys():
        print('{}: {} %'.format(key, importance[key]))


def train(x, y, clf, k, scoring, output_path=None):
    """ Trains a model and returns it
    :param x: Input data array of shape [n_samples, n_features]
    :param y: Label array of shape [n_samples]
    :param clf: Classifier
    :param k: Number of subsets to divide the training data into for cross validation
    :param scoring: scoring measure for k-fold cv
    :param output_path: Output path for model
    :return: model
    """
    clf.fit(x, y)
    score = cross_val_score(clf, x, y, cv=k, scoring=scoring)
    print('Score = {}, mean = {}'.format(score, score.mean()))
    try:
        print_feature_importance(clf)
    except AttributeError:
        print("Couldn't get feature importance for this classifier!")

    if output_path is not None:
        export_model(clf, output_path)

    return clf


def random_forest_grid():
    """ Return grid for RandomSearchCV for the Random Forest classifier

    :return: dictionary
    """
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = [i for i in range(1, 11)]
    max_depth = [i for i in range(1, 15)]
    min_samples_split = [i for i in range(2, 10)]
    min_samples_leaf = [i for i in range(1, 10)]

    grid = {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf}

    return grid


def random_forest_grid_fine():
    """ Return the finer grid for GridSearchCV for the Random Forest classifier

    :return: dictionary
    """
    n_estimators = [200]
    max_features = [2, 3, 4, 5]
    max_depth = [15, 16, 17]
    min_samples_split = [2, 3, 4, 5]
    min_samples_leaf = [1, 2, 3]

    grid = {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf}

    return grid


def svm_grid():
    """ Return grid for RandomSearchCV for the SVM classifier

    :return: dictionary
    """
    C = [0.1, 1, 10, 100]
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    degree = [2, 3, 4, 5, 6, 7, 8]
    coef0 = [0.1, 1, 10, 100]
    gamma = ['scale', 'auto']
    shrinking = [True, False]
    probability = [True, False]

    grid = {'C': C,
            'kernel': kernel,
            'degree': degree,
            'coef0': coef0,
            'gamma': gamma,
            'shrinking': shrinking,
            'probability': probability}

    return grid


def knn_grid():
    """ Return grid for RandomSearchCV for the KNN classifier

    :return: dictionary
    """
    n_neighbors = [i for i in range(1, 40)]
    weights = ['distance', 'uniform']
    p = [1, 2, 3, 4]

    grid = {'n_neighbors': n_neighbors,
            'weights': weights,
            'p': p}

    return grid


def find_optimal_classifier(clf, grid, x_train, y_train, k, scoring, iterations):
    """ Uses randomized search to find the best classifier.

    :param clf: classifier
    :param grid: grid for RandomSearchCV
    :param x_train: Training data input vectors
    :param y_train: Training data labels
    :param k: Number of subsets to divide the training data into for cross validation
    :param scoring: Scoring method
    :param iterations: number of random search iterations
    :return: Best model
    """
    search = RandomizedSearchCV(estimator=clf, param_distributions=grid, n_iter=iterations, cv=k, n_jobs=-1,
                                scoring=scoring, verbose=0)
    search.fit(x_train, y_train)

    print(f'Best params were:\n{search.best_params_}')
    print(f'Score was {search.best_score_}')

    return search.best_estimator_


def predict_words(words, clf, min_max_path):
    """ Makes prediction for a list of words and returns a list of positive labeled words.

    :param words: list of words
    :param clf: classifier
    :param min_max_path: path to folder with min max values
    :return: list of words
    """
    if wikipedia is None or documents is None:
        initialize_datasets()

    x_min = np.loadtxt(min_max_path + 'min_x')
    x_max = np.loadtxt(min_max_path + 'max_x')

    feature_vectors = get_feature_vector_of_list(words)
    for i in range(len(feature_vectors)):
        feature_vectors[i] = (feature_vectors[i] - x_min) / (x_max - x_min)
    predictions = clf.predict(feature_vectors)

    output_words = []
    for i in range(len(words)):
        if predictions[i] == 1:
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


def load_datasets(import_path=None, export_path=None, features=None):
    """ Load datasets, split them in training and validation sets and normalize the features.

    If import_path is specified load datasets from a file.
    If export_path is specified store datasets in files.

    :param import_path: Path to directory containing files to import training data from
    :param export_path: Path to directory to store files in
    :param features: List of features to choose
    :return: X_training, X_validation, y_training, y_validation
    """
    if features is None:
        features = list(range(11))
    if import_path is not None:
        x_train = np.loadtxt(f'{import_path}/x_train.csv', delimiter=',')[:, features]
        x_validate = np.loadtxt(f'{import_path}/x_validate.csv', delimiter=',')[:, features]
        y_train = np.loadtxt(f'{import_path}/y_train.csv', delimiter=',')
        y_validate = np.loadtxt(f'{import_path}/y_validate.csv', delimiter=',')

        return x_train, x_validate, y_train, y_validate
    else:
        positive_path = '../../output/training_data/training_small_positive_lemmas_labeled.txt'
        negative_small = '../../output/training_data/training_small_negative_labeled.txt'
        negative_medium = '../../output/training_data/training_medium_negative_labeled.txt'
        negative_large = '../../output/training_data/training_large_negative_labeled.txt'
        extra_data = '../../output/training_data/extra_data.txt'
        initialize_datasets()

        # get split datasets
        return get_data([positive_path, negative_large, extra_data], 0.3, export_path)


def evaluation(x_validate, y_validate, model):
    """ Evaluate a model and print the confusion matrix.

    :param x_validate: feature data of validation set
    :param y_validate: label data of validation set
    :param model: the model
    """
    y_predicted = model.predict(x_validate)
    cm = confusion_matrix(y_validate, y_predicted)
    plot_confusion_matrix(model, x_validate, y_validate)
    plt.savefig('../../output/plots/confusion_matrix.svg', format='svg')
    print('[[{}  {}]\n [{}  {}]]'.format(cm[0][0], cm[0][1], cm[1][0], cm[1][1]))
    print(f'F1 = {calc_f1(cm[1][1], cm[0][1], cm[1][0])}')
    print(f'Recall = {calc_recall(cm[1][1], cm[1][0])}')
    print(f'Precision = {calc_precision(cm[1][1], cm[0][1])}')


def get_best_classifiers(x_train, x_validate, y_train, y_validate, scoring, k):
    """ Get best classifiers by using randomized search on Random forest classifier, KNN and SVM classifier.

    :param x_train: Training data input vectors
    :param x_validate: Validation data input vectors
    :param y_train: Training data labels
    :param y_validate: Validation data labels
    :param k: Number of subsets to divide the training data into for cross validation
    :param scoring: Scoring method
    :return: best_rfc, best_knn, best_svm
    """
    start_time = timer()

    print('Find optimal RFC...')
    rfc = find_optimal_classifier(RandomForestClassifier(class_weight='balanced', n_jobs=-1), random_forest_grid(),
                                  x_train, y_train, k, scoring, iterations=300)
    evaluation(x_validate, y_validate, rfc)
    print(f'Done with search for RFC after {timer() - start_time}s.\n')
    start_time = timer()

    print('Find optimal KNN...')
    knn = find_optimal_classifier(KNeighborsClassifier(n_jobs=-1), knn_grid(),
                                  x_train, y_train, k, scoring, iterations=300)
    evaluation(x_validate, y_validate, knn)
    print(f'Done with search for KNN after {timer() - start_time}s.\n')
    start_time = timer()

    print('Find optimal SVM...')
    svmc = find_optimal_classifier(svm.SVC(class_weight='balanced'), svm_grid(),
                                   x_train, y_train, k, scoring, iterations=100)
    evaluation(x_validate, y_validate, svmc)
    print(f'Done with search for SVM after {timer() - start_time}s.\n')

    return rfc, knn, svmc


def rf_grid_search(x_train, y_train, k, scoring):
    """ Perform the grid search on the random forest.

    :param x_train: Training set input
    :param y_train: Training set labels
    :param k: Number of subsets to divide the training data into for cross validation
    :param scoring: scoring method
    :return: model
    """
    search = GridSearchCV(estimator=RandomForestClassifier(class_weight='balanced', n_jobs=-1),
                          param_grid=random_forest_grid_fine(),
                          cv=k, scoring=scoring, n_jobs=-1)
    search.fit(x_train, y_train)

    print(f'Best params were:\n{search.best_params_}')
    print(f'Score was {search.best_score_}')

    return search.best_estimator_


def main():
    # Load training data
    start_time = timer()
    x_train, x_validate, y_train, y_validate = load_datasets(import_path='datasets/default')
    print(f'Got data after {timer() - start_time}s')
    # set some parameters
    scoring = 'f1_macro'
    k = 6
    model_path = '../../output/models/random_forest.joblib'

    # randomized search
    # rfc, knn, svmc = get_best_classifiers(x_train, x_validate, y_train, y_validate, scoring, k)

    # grid search on rfc
    # rfc = rf_grid_search(x_train, y_train, k, scoring)
    # print(f'Grid search done after {timer() - start_time}')

    # these classifiers were estimated with the randomized search
    rfc = RandomForestClassifier(n_estimators=200, min_samples_split=3, min_samples_leaf=2, max_features=3,
                                 max_depth=17, bootstrap=True, class_weight='balanced', n_jobs=-1)
    knn = KNeighborsClassifier(weights='distance', p=1, n_neighbors=8, n_jobs=-1)
    svmc = svm.SVC(shrinking=True, probability=True, kernel='rbf', gamma='scale', degree=2, coef0=100, C=100,
                   class_weight='balanced')

    start_time = timer()
    rfc = train(x_train, y_train, rfc, k, scoring)
    print(f'Done with training for RFC after {timer() - start_time}s.\n')
    start_time = timer()
    evaluation(x_validate, y_validate, rfc)
    print(f'Done with evaluation for RFC after {timer() - start_time}s.\n')
    # store model
    dump(rfc, model_path)

    knn = train(x_train, y_train, knn, k, scoring)
    print(f'Done with training for KNN after {timer() - start_time}s.\n')
    start_time = timer()
    evaluation(x_validate, y_validate, knn)
    print(f'Done with evaluation for KNN after {timer() - start_time}s.\n')
    start_time = timer()

    svmc = train(x_train, y_train, svmc, k, scoring)
    print(f'Done with training for SVM after {timer() - start_time}s.\n')
    start_time = timer()
    evaluation(x_validate, y_validate, svmc)
    print(f'Done with evaluation for SVM after {timer() - start_time}s.\n')


if __name__ == '__main__':
    main()
