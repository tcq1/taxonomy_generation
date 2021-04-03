import numpy as np
import tensorflow.keras as keras


from keras import Sequential
from keras.layers import Dense, InputLayer, Dropout
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from sklearn.model_selection import RandomizedSearchCV

from timeit import default_timer as timer

from src.classification.classifier import load_datasets


input_shape = None


def get_model(x_train, y_train, w, l, optimizer, kernel_initializer, class_weights):
    """ Create, compile and train the mlp model.
    :param x_train: feature training data
    :param y_train: label training data
    :param w: width of hidden layers
    :param l: number of hidden layers
    :param optimizer: optimizer string
    :param kernel_initializer: initializer string
    :param class_weights: weights of label classes
    :return: trained model
    """
    model = Sequential()
    # add input layer
    model.add(InputLayer(input_shape))
    # add hidden layers
    for i in range(l):
        model.add(Dense(w, activation='relu', kernel_initializer=kernel_initializer))
    # add output layer
    model.add(Dense(2, activation='softmax'))
    # compile and fit model
    model.compile(optimizer=optimizer, loss=keras.losses.binary_crossentropy,
                  metrics=['accuracy'])
    # fit model using the training data
    start_time = timer()
    model.fit(x=x_train, y=y_train, batch_size=128, epochs=50, verbose=1, shuffle=True, class_weight=class_weights,
              use_multiprocessing=True)
    print(f'Done training after {timer() - start_time}')

    return model


def get_param_grid():
    """ Return the coarse parameter grid for the neural network.

    :return: dictionary of parameter grid
    """
    layer_width = [32, 64, 128, 256, 512]
    layers = [2, 3, 4, 5, 6]
    epochs = [10, 25, 50, 75, 100]
    batch_size = [32, 64, 96, 128, 160, 192, 224, 256]
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal',
                 'he_uniform']
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    optimizer = ['adam', 'sgd', 'adadelta', 'adagrad', 'adamax', 'ftrl', 'nadam', 'rmsprop']

    grid = {'layer_width': layer_width,
            'layers': layers,
            'epochs': epochs,
            'batch_size': batch_size,
            'activation': activation,
            'init_mode': init_mode,
            'dropout_rate': dropout_rate,
            'optimizer': optimizer}

    return grid


def create_model(layer_width=256, layers=3, activation='relu', epochs=50, batch_size=128,
                 init_mode='normal', dropout_rate=0, optimizer='adamax'):
    """ Build the model.

    :param layer_width: Width of hidden layers
    :param layers: number of hidden layers
    :param activation: activation function
    :param epochs: number of epochs
    :param batch_size: batch size
    :param init_mode: kernel initializer
    :param dropout_rate: dropout rate
    :param optimizer: optimizer
    :return: model
    """
    model = Sequential()
    # add input layer
    model.add(InputLayer(input_shape))
    # add hidden layers
    for layer in range(layers):
        model.add(Dense(layer_width, activation=activation, kernel_initializer=init_mode))
        # add dropout
        model.add(Dropout(dropout_rate))
    # add output layer
    model.add(Dense(2, activation='softmax'))
    # compile and fit model
    model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    return model


def get_best_model(x_train, y_train):
    """ This model was the best after grid search

    :param x_train: feature training data
    :param y_train: label training data
    :return: model
    """
    # calculate class weights
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train),
                                                      y_train)
    # convert to dict
    class_weights = dict(enumerate(class_weights))
    # encode label data
    y_train = to_categorical(y_train)

    return get_model(x_train, y_train, 256, 3, 'adamax', 'normal', class_weights)


def random_search(x_train, y_train, class_weights, iterations):
    """ Perform the randomized search.

    :param x_train: training features
    :param y_train: training labels
    :param class_weights: class weights
    :param iterations: number of iterations
    :return: model
    """
    # convert to dict
    class_weights = dict(enumerate(class_weights))
    model = KerasClassifier(build_fn=create_model)
    search = RandomizedSearchCV(estimator=model, param_distributions=get_param_grid(), n_jobs=-1,
                                cv=6, scoring='f1_macro', n_iter=iterations, verbose=1)
    search.fit(x_train, y_train, verbose=0, shuffle=True, class_weight=class_weights, batch_size=32, epochs=100)

    with open('best_model.txt', 'w') as f:
        f.write(str(search.best_params_))

    print(f'Best params were:\n{search.best_params_}')
    print(f'Score was {search.best_score_}')

    return search


def main():
    # get training and validation data
    x_train, x_validate, y_train, y_validate = load_datasets(import_path='datasets/default')

    global input_shape
    input_shape = (len(x_train[0]),)

    # calculate class weights
    # class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train),
    #                                                   y_train)

    # randomized search
    # model = random_search(x_train, y_train, class_weights, 100)

    model = get_best_model(x_train, y_train)

    print("Making predictions")
    start_time = timer()
    y_predict = model.predict(x_validate)
    print(f'Done prediction after {timer() - start_time}')

    print("Get confusion matrix")
    y_predict = np.argmax(y_predict, axis=1)
    cm = confusion_matrix(y_validate, y_predict)
    precision = cm[1][1] / (cm[1][1] + cm[0][1])
    recall = cm[1][1] / (cm[1][1] + cm[1][0])
    f1 = 2 * precision * recall / (precision + recall)
    print(f'F1-score = {f1}')
    print(f'[[{cm[0][0]}  {cm[0][1]}]\n [{cm[1][0]}  {cm[1][1]}]]')
    print(f"Done after {timer() - start_time}s.")


if __name__ == '__main__':
    main()
