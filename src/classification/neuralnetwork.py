import numpy as np
import tensorflow.keras as keras

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer
from keras.utils import to_categorical

from timeit import default_timer as timer

from src.classification.classifier import load_datasets


def get_model(x_train, y_train, w1, lmax):
    """ Create, compile and train the mlp model.
    :param x_train: feature training data
    :param y_train: label training data
    :param w1: width of first hidden layer
    :param lmax: number of hidden layers
    """
    input_shape = (len(x_train[0]),)
    widths = calc_width(w1, lmax)

    model = keras.Sequential()
    # add input layer
    model.add(InputLayer(input_shape))
    # add hidden layers
    for i in range(len(widths)):
        model.add(Dense(widths[i], activation='relu'))
    # add output layer, also tried this out with activation='relu' but result for unnormalized dataset became way worse
    model.add(Dense(2, activation='softmax'))
    # compile and fit model
    model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    # fit model using the training data
    model.fit(x=x_train, y=y_train, batch_size=32, epochs=100, verbose=0, shuffle=True)

    return model


def calc_width(initial_w, l):
    """ Calculate the width for all hidden layers.
    :param initial_w: width of first layer
    :param l: number of hidden layers
    """
    widths = [initial_w]
    for i in range(1, l):
        widths.append(int(widths[i-1] / 2))

    return widths


def grid_search(x_train, y_train, x_validate, y_validate):
    """ Perform a grid search to find optimal values for w1 and lmax.
    :param x_train: feature training data
    :param y_train: label training data
    :param x_validate: feature validation data
    :param y_validate: label validation data
    :return: ((w1_best, lmax_best), (mse))
    """
    # define the grid
    w_values = [1024, 512, 256, 128, 64, 32]
    l_values = [2, 3, 4, 5, 6, 7]

    # store trained models in here
    models = {}

    # grid search
    for w in w_values:
        for l in l_values:
            keras.backend.clear_session()
            # get trained model
            model = get_model(x_train, y_train, w, l)
            # evaluate and store params and score in models dictionary
            models[(w, l)] = model.evaluate(x=x_validate, y=y_validate)

    # print params and score of all models
    for k, v in models.items():
        print("{}: {}".format(k, v))

    # find best model
    best = min(models.items(), key=lambda x: x[1])

    return best


def main():
    # get training and validation data
    x_train, x_validate, y_train, y_validate = load_datasets()
    y_train = to_categorical(y_train)
    y_validate = to_categorical(y_validate)

    start_time = timer()
    print("---------- Grid search on normalized dataset ----------")
    # for testing purposes the grid search was iterated 5 times, set this to 1 to not iterate the grid search
    iterations = 1
    best_list = []
    for i in range(iterations):
        best_list.append(grid_search(x_train, y_train, x_validate, y_validate))
    for i in range(iterations):
        print("Best parameters were w1 = {}, lmax = {}. MSE of model was {}".format(best_list[i][0][0],
                                                                                    best_list[i][0][1],
                                                                                    best_list[i][1]))
    print(f"Done after {timer() - start_time}s.")


if __name__ == '__main__':
    main()
