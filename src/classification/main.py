from sklearn import tree
from sklearn.model_selection import train_test_split
from src.classification.extract_features import *
import graphviz
from sklearn.metrics import accuracy_score


def extract_feature_vector(word):
    return [has_capital_letter(word), contains_hyphen(word), get_word_length(word), get_number_syllables(word)]


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


def main():
    positive_path = '../../output/training_small_positive.txt'
    negative_path = '../../output/training_small_negative.txt'

    positive_data, labels = load_data(positive_path, 1)
    negative_data, labels2 = load_data(negative_path, 0)

    training_data = positive_data
    training_data.extend(negative_data)
    labels.extend(labels2)

    X_train, X_test, y_train, y_test = train_test_split(extract_feature_vector_of_list(training_data), labels,
                                                        test_size=0.3, random_state=1)

    feature_names = ['Has capital letter', 'Contains hyphen', 'Word length', 'Number syllables']
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    tree.plot_tree(clf)

    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_names,
                                    filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)

    print(accuracy_score(y_test, clf.predict(X_test)))
    graph.view()


if __name__ == '__main__':
    main()
