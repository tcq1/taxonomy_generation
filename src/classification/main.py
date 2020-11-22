from sklearn import tree
from sklearn.model_selection import train_test_split
from src.classification.extract_features import *
import graphviz
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from src.text_extraction.csv_manager import import_docs
import datetime


def extract_feature_vector(word, documents):
    return [has_capital_letter(word), contains_hyphen(word), get_word_length(word), get_number_syllables(word),
            appearance_per_doc_length(word, documents)[0], appearance_per_doc_length(word, documents)[1]]


def extract_feature_vector_of_list(words, documents):
    features = []

    for word in words:
        features.append(extract_feature_vector(word, documents))

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


def main():
    positive_path = '../../output/training_small_positive.txt'
    negative_path = '../../output/training_small_negative.txt'

    positive_data, labels = load_data(positive_path, 1)
    negative_data, labels2 = load_data(negative_path, 0)

    documents_path = '../../output/dictionary_lemmas.csv'
    documents = import_docs(documents_path)

    training_data = positive_data
    training_data.extend(negative_data)
    labels.extend(labels2)

    # x_train, x_test, y_train, y_test = train_test_split(extract_feature_vector_of_list(training_data, documents),
    #                                                     labels, test_size=0.3, random_state=1)

    # # Decision tree
    # clf = tree.DecisionTreeClassifier()
    # clf.fit(x_train, y_train)

    start = datetime.datetime.now()

    clf = RandomForestClassifier(n_estimators=1000, max_features=None, max_depth=6, min_samples_split=2, n_jobs=8)
    clf.fit(extract_feature_vector_of_list(training_data, documents), labels)
    score = cross_val_score(clf, extract_feature_vector_of_list(training_data, documents), labels, cv=10)
    print('Score = {}, mean = {}'.format(score, score.mean()))
    print('Importances: {}'.format(clf.feature_importances_))
    end = datetime.datetime.now()
    print('Total time for training: {}'.format(end-start))

    # print(accuracy_score(y_test, clf.predict(x_test)))

    # feature_names = ['Has capital letter', 'Contains hyphen', 'Word length', 'Number syllables', 'Appearance ratio',
    #                  'Appearances in docs']

    # tree.plot_tree(clf)
    # dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_names,
    #                                 filled=True, rounded=True, special_characters=True)
    # graph = graphviz.Source(dot_data)
    # graph.view()


if __name__ == '__main__':
    main()
