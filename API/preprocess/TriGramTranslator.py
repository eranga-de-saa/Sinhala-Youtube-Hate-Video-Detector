import pickle
import nltk
from preprocess.translator import convertText

translateorPic = open("preprocess/trigramTrans.pickle", "rb")
translator = pickle.load(translateorPic)
def triGramTranslate(sentence):
    translation = ""
    translated = translator.tag(nltk.word_tokenize(sentence.lower()))
    for word, trans in translated:
        if trans in ('NNN'):
            translation = translation+str(convertText(word)+" ")
        else:
            translation = translation+str(trans+" ")
    return translation