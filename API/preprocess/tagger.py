from nltk.tag import str2tuple, tuple2str
import pickle
import nltk

tagger_t = open("preprocess/perceptron_tagger.pickle", "rb")
tagger_save = pickle.load(tagger_t)


def untagging(string):
    untag = ""
    for t in string:
        untag = untag + tuple2str(t) + " "
    return untag


def tagg(sentence):
    print(sentence)
    tagged = tagger_save.tag(nltk.word_tokenize(sentence))
    return untagging(tagged)
