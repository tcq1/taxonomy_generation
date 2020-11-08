import pyphen


def get_number_syllables(word):
    dic = pyphen.Pyphen(lang='de_DE')
    split_word = dic.inserted(word).split('-')

    # if '-' appears in word then there will be empty strings in split_word --> remove empty strings
    while '' in split_word:
        split_word.remove('')

    return len(split_word)


def get_word_length(word):
    return len(word)


def has_capital_letter(word):
    return word[0].isupper()
