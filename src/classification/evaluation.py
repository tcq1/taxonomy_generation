import matplotlib.pyplot as plt
import numpy as np


def plot_scores(output_path, colors):
    """ Make bar plot of the score metrics and save image in output_path folder.

    :param output_path: output path
    :param colors: list of colors
    """
    # data
    rf_scores = [0.628099173553719, 0.7755102040816326, 0.5277777777777778, 0.9735025606769093]
    svm_scores = [0.3546284224250326, 0.9251700680272109, 0.21935483870967742, 0.8915187376725838]
    knn_scores = [0.4936170212765957, 0.3945578231292517, 0.6590909090909091, 0.9739206662283585]
    ann_scores = [0.4425956738768719, 0.9047619047619048, 0.29295154185022027, 0.9265833881218497]

    # width of bar
    width = 0.2

    # set x positions of bars
    rf_pos = np.arange(len(rf_scores))
    svm_pos = [x + width for x in rf_pos]
    knn_pos = [x + width for x in svm_pos]
    ann_pos = [x + width for x in knn_pos]

    # Make the plot
    plt.bar(rf_pos, rf_scores, color=colors[0], width=width, edgecolor='white', label='RF')
    plt.bar(svm_pos, svm_scores, color=colors[1], width=width, edgecolor='white', label='SVM')
    plt.bar(knn_pos, knn_scores, color=colors[2], width=width, edgecolor='white', label='KNN')
    plt.bar(ann_pos, ann_scores, color=colors[3], width=width, edgecolor='white', label='ANN')

    # Add xticks on the middle of the group bars
    plt.xlabel('Metrics')
    plt.xticks([r + width for r in range(len(rf_scores))], ['F1', 'Recall', 'Precision', 'Accuracy'])

    # Create legend & Show graphic
    plt.legend(loc='right', bbox_to_anchor=(1, 0.5))
    plt.savefig(output_path + 'scores.svg', format='svg')
    plt.close()


def plot_best_rf_model(output_path, colors):
    """ Make bar plot of the training times and save image in output_path folder.

    :param output_path: output path
    :param colors: list of colors
    """
    # data
    metrics = ['F1', 'Recall', 'Precision', 'Accuracy']
    data_label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    scores = [[0.6548, 0.7483, 0.5820, 0.9750],
              [0.6440, 0.7075, 0.5909, 0.9748],
              [0.6154, 0.6803, 0.5618, 0.9726],
              [0.6300, 0.7007, 0.5722, 0.9735],
              [0.5989, 0.7211, 0.5121, 0.9689],
              [0.5604, 0.7415, 0.4504, 0.9625],
              [0.4410, 0.8776, 0.2945, 0.9283],
              [0.2743, 0.2109, 0.3924, 0.9641]]

    # width of bar
    width = 0.1

    # set x positions
    pos = [np.arange(len(metrics))]
    for i in range(1, len(data_label)):
        pos.append([x + width for x in pos[i-1]])

    # plot bars
    for label in range(len(data_label)):
        plt.bar(pos[label], scores[label], color=colors[label], width=width, edgecolor='white', label=data_label[label])

    # Add xticks on the middle of the group bars
    plt.xlabel('Metrics')
    plt.xticks([r + 3.5*width for r in range(len(metrics))], metrics)

    # Create legend & Show graphic
    plt.legend(loc='right', bbox_to_anchor=(1.2, 0.5))
    plt.savefig(output_path + 'best_rf.svg', format='svg')
    plt.close()


def plot_time(output_path, file_name, colors, times, ylabel):
    """ Make bar plot of the training times and save image in output_path folder.

    :param output_path: output path
    :param file_name: file name
    :param colors: list of colors
    :param times: list of times
    :param ylabel: y label
    """
    # data
    labels = ['RF', 'SVM', 'KNN', 'ANN']

    x_pos = [i for i, _ in enumerate(labels)]
    plt.bar(x_pos, times, color=colors)
    plt.xlabel('Models')
    plt.ylabel(ylabel)
    plt.xticks(x_pos, labels)

    plt.legend()
    plt.savefig(output_path + file_name + '.svg', format='svg')
    plt.close()


def calc_recall(tp, fn):
    return tp / (tp + fn)


def calc_precision(tp, fp):
    return tp / (tp + fp)


def calc_f1(tp, fp, fn):
    return 2 * calc_recall(tp, fn) * calc_precision(tp, fp) / (calc_recall(tp, fn) + calc_precision(tp, fp))


def calc_accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def main():
    output_path = '../../output/plots/'

    training_time = [17.6715169, 331.77303390000003, 0.6349119999999999, 6.5790526]
    prediction_time = [0.6124921000000008, 0.05822790000001987, 0.21460550000000111, 0.0629135999999999]

    colors = ['#306182', '#768b99', '#5ebeff', '#9c5959', '#93827f', '#2F2F2F', '#92B4A7', '#545E56']

    plot_best_rf_model(output_path, colors)
    plot_scores(output_path, colors)
    plot_time(output_path, 'training_time', colors, training_time, 'Training time in s')
    plot_time(output_path, 'prediction_time', colors, prediction_time, 'Prediction time in s')


if __name__ == '__main__':
    main()
