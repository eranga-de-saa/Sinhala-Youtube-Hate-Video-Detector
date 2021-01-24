import nltk
import re
import string
# from googletrans import Translator
import emoji
from preprocess.tagger import tagg
from preprocess.stemmer import stemm_replace
from preprocess.translator import convertText, translate_singlish_sinhala
from preprocess.eng_translator import translate_english_sinhala
from preprocess.TriGramTranslator import triGramTranslate
import fasttext
import http.client
import urllib.parse
import urllib.request
# gTrans = Translator()
import json
# import os
# os.system("tagger.py")
# os.system("stemmer.py")
# os.system("translator.py")
text_model = fasttext.load_model("preprocess/lid.176.bin")

def removeEmoji(text):
    allchars = [str for str in text]
    for word in allchars:
        if word in emoji.UNICODE_EMOJI:
            emoji_desc = emoji.demojize(word)
            text = text.replace(word, " ")
    return text

def remove_url_punctuations(comment):
    url_removed = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%|\-)*\b', ' ', comment)
    removed = re.sub("[" + re.escape(string.punctuation) + "]", " ", url_removed)
    return removed

# def remove_url(comment):
    # url_removed = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%|\-)*\b', ' ', comment)
    # return url_removed


def tokenizeLine(line):
    tokens = nltk.word_tokenize(line)
    return tokens


def googleTranslate(line):
        sourceLang = "en"
        targetLang = "si"
        conn = urllib.request.urlopen("https://translate.googleapis.com/translate_a/single?client=gtx&sl="
                                      + sourceLang + "&tl=" + targetLang + "&dt=t&q=" + urllib.parse.quote(line))
        raw_data = conn.read()
        encoding = conn.info().get_content_charset('utf8')  # JSON default
        data = json.loads(raw_data.decode(encoding))
        return data[0][0][0]

#
# def remove_punctuations(text):
#     without_punc = "".join([c for c in text if c not in string.punctuation])
#     return without_punc


def process(line):
    # print(line)
    line_url = remove_url_punctuations(line)
    line_emogi = removeEmoji(line_url)
    text = line_emogi
    tkline = tokenizeLine(line)
    cleanText = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)


    if u'\u0D80' <= cleanText.strip() <= u'\u0DFF' and text is not None:
        textTrans = translate_english_sinhala(text.lower())
        tagged = tagg(textTrans)
        sentence = stemm_replace(tagged)
    elif re.match('[a-zA-Z]', cleanText.strip()):
        text1 = text_model.predict(text.lower().replace("\n", " "))
        if "en" in str(text1[0]) and float(str(text1[1]).replace("[", "").replace("]", "")) > 0.70:
            try:
                text = googleTranslate(text)
                tagged = tagg(text)
                sentence = stemm_replace(tagged)
            except:
                print("Translating in progress ....")
                text = translate_english_sinhala(text.lower())
                tagged = tagg(text)
                sentence = stemm_replace(tagged)
            # sentence = "NONE"

        elif 1==2:
            sentence = "NONE"
        else:
            # if str(detect.lang) == "ta" and detect.confidence > 0.70:
            #     sentence = None
            # else:
            text = triGramTranslate(text.lower())
            tagged = tagg(text)
            sentence = stemm_replace(tagged)
    else:
        sentence = "NONE"

    return sentence

# print(remove_url(https://www.youtube.com/watch?v=FgSVHKTEI-8&feature=youtu.be))
# print(process("Apiwath sub karala ekathu wenna"))
# print(process("මොන හුත්තක්ද මේ"))