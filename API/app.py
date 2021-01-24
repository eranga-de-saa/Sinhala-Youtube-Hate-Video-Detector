from flask import Flask, url_for, request, jsonify
import os
from flask_cors import CORS
from sklearn.externals import joblib
import nltk
import random
import json
from pandas.io.json import json_normalize
from pprint import pprint
from collections import defaultdict
import pandas as pd
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import string
import random
from sklearn.ensemble import VotingClassifier
import pickle
import os
import urllib.request
from urllib.parse import urlparse, urlencode, parse_qs
from youtube_data import VideoData
from random import seed
from random import randint
from thumbnail import img_preprocess as img_pr
from preprocess import preprocessor as pr
import multiprocessing as mp
import tensorflow as tf
import keras
import shutil
from decimal import Decimal

app = Flask(__name__)
CORS(app)
# cors = CORS(app, resources={r"/*": {"origins": "*"}})

political_words = pd.read_excel(r'models/PVocabulary.xlsx')

relious_ethnic_words = pd.read_excel(r'models/REVocabulary.xlsx')

sex_gender_words = pd.read_excel(r'models/SGVocabulary.xlsx')

political_vocabulary = [word.strip() for word in political_words['words']]

relious_ethnic_vocabulary = [word.strip() for word in relious_ethnic_words['words']]

sex_gender_vocabulary = [word.strip() for word in sex_gender_words['words']]

df_positive_words = pd.read_excel(r'models/positivewords.xlsx')

df_negative_words = pd.read_excel(r'models\HateWords.xlsx')

negation_words = {"නැ", "නෑ", "නැහැ", "නැත", "නැති", "බැ", "බෑ", "බැහැ", "බැරිය", "එපා"}

comment_sentiModel = joblib.load("models/comment_Lr.pkl")
comment_feature_mapper = joblib.load("models/comment_feature_mapper.pkl")

positive_vocabulary = [word.strip() for word in df_positive_words['words']]
negative_vocabulary = [word.strip() for word in df_negative_words['words']]


def imgprocess():
    thumbnail_text = img_pr.process()
    # keras.backend.clear_session()
    folder = 'img_folder'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return thumbnail_text


@app.route('/')
def index():
    str = "Hello World"
    return jsonify(message=str)



@app.route('/image_process', methods=['GET'])
def image_process():
    #req_data = request.get_json()
    #print(req_data)
    # video_url = req_data['url']
    vid=request.args.get("url")
    # vid= req_data['url']
    print(vid)
    # seed(42)

    keys = pd.read_csv(os.getcwd() + "\\keys.csv", encoding='utf-')
    # url = video_url
    noOfKeys = keys.shape[0]
    print(noOfKeys)
    # print(noOfKeys)

    # video_id = urlparse(url)
    # q = parse_qs(video_id.query)
    # vid = q["v"][0]
    key = keys.iloc[randint(0, noOfKeys - 1), 0]
    print("_______________________________________________")
    print(key)
    # try:
    vc = VideoData(vid, key)
    vc.get_video_comments()
    # except Exception:
    # print("Missing Meta Data")
        # return 0
    urllib.request.urlretrieve("https://img.youtube.com/vi/" + vid + "/hqdefault.jpg",
                               "temp/img.jpg")
    thumbnail_text = imgprocess()
    # thumbnail_text = "Something"
    print("Number of processors: ", mp.cpu_count())
    pool = mp.Pool(6)  # for parallel processing
    df = pd.DataFrame(
        columns=['comment', 'otherMetadata', 'likeDislikeRatio', 'posToNegCommentRatio',
                 'pcount', 'recount', 'sgcount'])

    with open("temp/data.json", encoding='utf-8') as f:
        data = json.load(f)
        pcount = 0
        recount = 0
        sgcount = 0
        wordcount = 0
        # global commentsText
        commentsText = ""
        # tags = data[0]["tags"]
        if 'tags' in data[0].keys():
            tags = str(' '.join(data[0]["tags"]))
            # print("tags" + tags)
            otherMetaData = pr.process(data[0]["title"] + " " + data[0]["description"] + " " + tags + " " + thumbnail_text)
        else:
            otherMetaData = pr.process(data[0]["title"] + " " + data[0]["description"] + " " + thumbnail_text)

        likes = int(data[0]["likeCount"])
        dislikes = int(data[0]["dislikeCount"])
        likeDislikeRatio = str(float(likes / dislikes))
        results = pool.map(preprocess, [item for item in data[0]["comments"]])
        pool.close()
        postiveCount = 1
        negativeCount = 1
        for result in results:
            if result[0] is not None:
                commentsText = commentsText + " " + result[0]
                if result[1] == 1:
                    postiveCount = postiveCount + 1
                else:
                    negativeCount = negativeCount + 1

            if result[2] is not None:
                pcount = pcount + int(result[2])

            if result[3] is not None:
                recount = recount + int(result[3])

            if result[4] is not None:
                sgcount = sgcount + int(result[4])

            if result[5] is not None:
                wordcount = wordcount + int(result[5])

        p_count = str(float((pcount * 100 / wordcount)))
        re_count = str(float((recount * 100 / wordcount)))
        sg_count = str(float((sgcount * 100 / wordcount)))

        posToNegCommentRatio = str(float(postiveCount / negativeCount))
        df.loc[0] = [commentsText] + [otherMetaData] + [likeDislikeRatio] + [posToNegCommentRatio] + [p_count] + [
            re_count] + [sg_count]

        # print(df['otherMetadata'].iloc[0])

        # Classify
        # Classification_Model = joblib.load("classification_Lr.pkl")
        # Classification_vectorizer  = joblib.load("vectorizer_mapper.pkl")
        #
        # tranformed = Classification_vectorizer.transform(df)
        # print(tranformed[0][2])
        # prediction = Classification_Model.predict(tranformed)
        # print(prediction)

        keras.backend.clear_session()

        HateDetection_feature_mapper = joblib.load("models/HD_featureMapper_LR.pkl")
        HateDetection_Model = joblib.load("models/HateDetection_LR.pkl")

        tranformed = HateDetection_feature_mapper.transform(df)
        hate = HateDetection_Model.predict(tranformed)
        print("Hate Detection :" + str(hate))
        clf = HateDetection_Model
        z = np.dot(clf.coef_, tranformed.T) + clf.intercept_
        hypo = 1 / (1 + np.exp(-z))
        hate_level = 1 - float(hypo)
        print("Level of hate =" + str(hate_level))
        print(hate_level)

        df['data'] = df['comment'] + " " + df['otherMetadata']

        keras.backend.clear_session()

        Domain_feature_mapper = joblib.load("models/Domain_feature_mapper.pkl")

        Domain_Classification_Model = tf.keras.models.load_model('models/Domain_ann.pkl')

        Domaintranformed = Domain_feature_mapper.transform(df)
        Domain = Domain_Classification_Model.predict_classes(Domaintranformed)

        Domain_label_encoder = joblib.load("models/Domain_label_encoder.pkl")

        domain_label = Domain_label_encoder.inverse_transform(Domain)

        print("Domain :" + str(domain_label))

        print(domain_label)
        print(hate_level)

        if domain_label[0] == 'P':
            domain = 'Political'
        elif domain_label[0] == 'SG':
            domain = 'Sex & Gender'
        elif domain_label[0] == 'RE':
            domain = 'Religious & Ethnic'
        else:
            domain = 'Other'

        if hate[0] == 1:
            sentiment = 'Not Hate'
             # sentiment = 'Low Hate'
        else:
            sentiment = 'Hate'
            # sentiment = 'High Hate'

        x = Decimal(hate_level)
        level_of_hate = float(round(x, 2))
        print(level_of_hate)

        return jsonify(sentiment=sentiment,
                       hateLevel=level_of_hate,
                       category=domain,
                       thumbnail_text=thumbnail_text)

    # return jsonify(thumbnail_text=thumbnail_text)


@app.route('/pre_process', methods=['POST'])
def pre_process():
    req_data = request.get_json()

    video_url = req_data['url']
    thumbnail_text = req_data['url']
    return jsonify(success="success")


@app.route('/text_process', methods=['GET'])
def text_process():
    sentiment = "Hate"
    hate_level = 0.7
    category = "Political"

    return jsonify(sentiment=sentiment,
                   hateLevel=hate_level,
                   category=category)




# def preprocess(item):
#     """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
#     comment = str(item["comment"])
#     processed_comment = pr.process(comment)
#     print(processed_comment)
#     if (processed_comment != "None") and (processed_comment is not None) and (processed_comment != ""):
#         # transformed = comment_feature_mapper.transform([processed_comment])
#         # sentiment = int(comment_sentiModel.predict(transformed))
#
#         PCount = 0
#         RECount = 0
#         SGCount = 0
#         wordCount = 0
#         positive_count = 0
#         negative_count = 0
#
#         for w in processed_comment.split():
#             wordCount = wordCount + 1
#             word = w
#             if "/" in word:
#                 word = w[:w.index('/')]
#             if word in positive_vocabulary:
#                 positive_count = positive_count + 1
#             elif word in negative_vocabulary:
#                 negative_count = negative_count + 1
#             elif word in political_vocabulary:
#                 PCount = PCount + 1
#             elif word in relious_ethnic_vocabulary:
#                 RECount = RECount + 1
#             elif word in sex_gender_vocabulary:
#                 SGCount = SGCount + 1
#
#         positive_count = positive_count / wordCount
#         negative_count = negative_count / wordCount
#
#         data = {'preprocessed_text': [processed_comment], 'positive_count': [positive_count],
#                 'Negative_count': [negative_count]}
#
#         comment_df = pd.DataFrame(data)
#
#         transformed = comment_feature_mapper.transform(comment_df)
#         sentiment = int(comment_sentiModel.predict(transformed))
#
#         print(PCount, RECount, SGCount, wordCount)
#         return processed_comment, sentiment, PCount, RECount, SGCount, wordCount
#         # return processed_comment, sentiment
#     else:
#         return None, None, None, None, None, None
#         # return None, None

def preprocess(item):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    comment = str(item["comment"])
    processed_comment = pr.process(comment)
    print(processed_comment)
    if (processed_comment != "None") and (processed_comment is not None) and (processed_comment != ""):
        # transformed = comment_feature_mapper.transform([processed_comment])
        # sentiment = int(comment_sentiModel.predict(transformed))

        PCount = 0
        RECount = 0
        SGCount = 0
        wordCount=0
        positive_count = 0
        negative_count = 0
        comment_neg = processed_comment
        for i, w in enumerate(processed_comment.split()):
            wordCount = wordCount + 1
            word = w
            if "/" in word:
                word = w[:w.index('/')]
            if word in positive_vocabulary:
                positive_count = positive_count + 1
            elif word in negative_vocabulary:
                negative_count = negative_count + 1
            elif word in political_vocabulary:
                PCount = PCount + 1
            elif word in relious_ethnic_vocabulary:
                RECount = RECount + 1
            elif word in sex_gender_vocabulary:
                SGCount = SGCount + 1
            elif word in negation_words:
                print("comment :" + comment)
                word = processed_comment.split()[i]
                print("Negation word :" + word)
                previous_word = word = processed_comment.split()[i - 1]
                new_previous_word = "not_" + previous_word
                print("Previous Word : " + previous_word)
                before_previous_word = processed_comment[:processed_comment.find(previous_word)]
                # print("Before Previous Word: " + before_previous_word)
                after_previous_word = processed_comment[processed_comment.find(previous_word) + len(previous_word):]
                # print("After previous Word:" + after_previous_word)
                comment_neg = before_previous_word + ' ' + new_previous_word + ' ' + after_previous_word
        positive_count = positive_count / wordCount
        negative_count = negative_count / wordCount

        data = {'preprocessed_text': [comment_neg], 'positive_count': [positive_count],
                'Negative_count': [negative_count]}

        comment_df = pd.DataFrame(data)

        transformed = comment_feature_mapper.transform(comment_df)
        sentiment = int(comment_sentiModel.predict(transformed))

        print(PCount, RECount, SGCount, wordCount)
        return processed_comment, sentiment, PCount, RECount, SGCount, wordCount
        # return processed_comment, sentiment
    else:
        return None,None,None,None,None,None
        # return None, None

if __name__ == "__main__":
    app.run(debug=True)
