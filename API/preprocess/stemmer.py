import functools
import operator
import re
import nltk

stem_dictionary = "preprocess/stem_dic.txt"
stopWordsPath = "preprocess/StopWords.txt"


def convert_tuples(tup):
    str = functools.reduce(operator.add, (tup))
    return str


# Get the sinhala dictionary
stem_dict = [(l.strip(), 'utf-8')
             for l in open(stem_dictionary, encoding='utf-8')]

stem_dictionary = {}
for s in stem_dict:
    s = ''.join(s)
    s = s.split("\t")
    s[1] = s[1].strip("utf-8")
    stem_dictionary[s[0]] = s[1]

# stopwords
stopwords = [(l.strip(), 'utf-8')
             for l in open(stopWordsPath, encoding='utf-16')]

StopWordsDic = []
for word in stopwords:
    word = ''.join(word)
    word = word.split("\t")
    StopWordsDic.append(word[0])


def stemm_replace(text):
    tokens = nltk.word_tokenize(text)
    regex = re.compile(u'[^\u0D80-\u0DFF]', re.UNICODE)
    Stokens = [regex.sub('', w) for w in tokens]

    for k, v in enumerate(Stokens):
        text = str(text).replace(Stokens[k], stem_dictionary.get(v, v))
        if re.match('^[0-9]*$', Stokens[k]):
            text = str(text).replace(tokens[k], "")
        elif Stokens[k] in StopWordsDic:
            text = str(text).replace(tokens[k], "")

    if (text.strip()==""):
        text = None

    return text
