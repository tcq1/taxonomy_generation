from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow

from joblib import load

from src.classification.classifier import predict_words

import spacy

# load spacy nlp model
nlp = spacy.load('de_core_news_lg')

# load trained ml model
model_path = '../../output/models/random_forest.joblib'
model = load(model_path)


class AppWindow(QMainWindow):
    def __init__(self):
        super(AppWindow, self).__init__()
        # layouts
        self.horizontal_layout = QtWidgets.QHBoxLayout()
        self.right_vertical_layout = QtWidgets.QVBoxLayout()

        # widgets
        self.input_text = QtWidgets.QTextEdit()
        self.output_text = QtWidgets.QTextEdit()
        self.output_word_list = QtWidgets.QTextEdit()
        self.button = QtWidgets.QPushButton()
        self.widget = QtWidgets.QWidget()

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Finde Wörter aus dem Werkstoffprüfungskontext")
        self.setGeometry(500, 500, 500, 500)

        # setup input text
        self.input_text.setPlaceholderText("Geben Sie einen Text oder eine Liste von Wörtern ein!\n\n"
                                           "Falls Sie eine Liste von Wörtern eingeben: es macht keinen Unterschied wie "
                                           "die Wörter voneinander getrennt werden.")
        self.input_text.textChanged.connect(self.input_field_value_change)

        # setup output text
        self.output_text.setReadOnly(True)

        # setup output word list
        self.output_word_list.setReadOnly(True)

        # add both output widgets to vertical layout
        self.right_vertical_layout.addWidget(self.output_text)
        self.right_vertical_layout.addWidget(self.output_word_list)

        # setup button
        self.button.setText("Finde Wörter")
        self.button.setEnabled(False)
        self.button.clicked.connect(self.button_clicked)

        # add widgets
        self.horizontal_layout.addWidget(self.input_text)
        self.horizontal_layout.addWidget(self.button)
        self.horizontal_layout.addLayout(self.right_vertical_layout)

        # add layout to window
        self.widget.setLayout(self.horizontal_layout)
        self.setCentralWidget(self.widget)

    def button_clicked(self):
        """ Split up text in tokens and detect words.
        """
        # get input text
        text = self.input_text.toPlainText()

        # get words that need to be highlighted
        detected_words = get_positive_words(text)
        # highlight words
        for word in detected_words:
            text = text.replace(word, highlight_text(word))

        # set output text
        self.output_text.setText(text)

        # set output word list text
        word_list_text = ""
        for word in detected_words:
            word_list_text += word + "\n"
        self.output_word_list.setText(word_list_text)

    def input_field_value_change(self):
        """ Value change listener for the input field. Set button enabled or disabled depending on whether the
        text edit is empty or not.
        """
        if self.input_text.toPlainText() == "":
            self.button.setEnabled(False)
            self.input_text.setPlaceholderText("Geben Sie einen Text oder eine Liste von Wörtern ein!\n\n"
                                               "Falls Sie eine Liste von Wörtern eingeben: es macht keinen "
                                               "Unterschied wie die Wörter voneinander getrennt werden.")
        else:
            self.button.setEnabled(True)


def get_positive_words(text):
    """ Find all positively labeled words in text and return a list with them.

    :param text: string
    :return: list of words
    """
    # find all tokens that contain letters
    words = [token.text for token in nlp(text) if 'x' in token.shape_.lower()]

    return predict_words(words, model)


def highlight_text(text):
    """ Wrap HTML styling around a text to make it red

    :param text: string
    :return: string with HTML tags
    """
    return "<span style=\" font-size:8pt; font-weight:600; color:#ff0000;\" >" + text + "</span>"


def main():
    # run the application
    app = QApplication([])
    window = AppWindow()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
