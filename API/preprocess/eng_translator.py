import re
from preprocess.translator import convertText

dictionary = {}
with open('preprocess/en-si.txt', 'r', encoding="utf8") as file:
    for line in file:
        key, value = line.strip().split("=")
        dictionary[key] = value.split("|")[0]


def translate_english_sinhala(x):
    if re.search('[a-zA-Z]', x)is not None:
        # print(x)
        for word in x.split():
            if re.match('[a-zA-Z]', word) is not None:
                word_lower = word.lower()
                translated_word = dictionary.get(word_lower)
                if translated_word is None:
                    translated_word = convertText(word)
                x = x.replace(word, translated_word)
    return x