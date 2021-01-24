import codecs
import glob
import os
import re
import cv2
import keras
import numpy as np
import tensorflow as tf
import thumbnail.spell_correction as sc
from keras import backend as k



# saved_model = tf.keras.models.load_model('model_cnn_identify.pkl')
# # saved_model = tf.keras.models.load_model('model_ann_identify.pkl')
# img_no = '29'
# images = []
# indexes = []
# path = os.path.join(img_no, '*g')
# files = glob.glob(path)
# numbers = re.compile(r'(\d+)')


class prediction:
    def __init__(self, img_no):
        self.saved_model = tf.keras.models.load_model('thumbnail/model_cnn_identify.pkl')
        # self.saved_model._make_predict_function()
        self.graph = tf.get_default_graph()
        # saved_model = tf.keras.models.load_model('model_ann_identify.pkl')
        self.img_no = img_no
        self.images = []
        self.indexes = []
        self.path = os.path.join(self.img_no, '*g')
        self.files = glob.glob(self.path)
        self.numbers = re.compile(r'(\d+)')

    def numericalsort(self, value):
        parts = self.numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    def identifyprocess(self):
        i = 0
        for fl in sorted(self.files, key=self.numericalsort):
            image = cv2.imread(fl, 0)
            image = cv2.resize(image, (50, 50))
            # image = invert(image)
            image = np.reshape(image, (50, 50, 1))
            self.images.append(image)
            if os.path.splitext(fl.title())[0][-1] == '1' and os.path.splitext(fl.title())[0][-2] != '1':
                self.indexes.append(i)
            i = i + 1

        # print(indexes[1:])
        x = np.array(self.images)
        print(len(x))
        with self.graph.as_default():
            classes = self.saved_model.predict_classes(x)
        print(classes)
        k.clear_session()

        class_file = codecs.open("thumbnail/wijesekara_map.txt", "r", "utf-8-sig")
        uni_classes = class_file.read().split("\t")
        class_file.close()

        mapped_classes = []

        j = 0
        for cl in classes:
            if (self.indexes[1:].__contains__(j)):
                mapped_classes.append(" ")
            mapped_classes.append(uni_classes[cl - 1])
            j = j + 1

        print(mapped_classes)

        sentence_predicted = ''.join(mapped_classes)
        print(sentence_predicted)
        sentence_predicted = sentence_predicted.replace("$", "").replace("   ", " ").replace("  ", " ")
        if sentence_predicted and sentence_predicted[0] == " ":
            sentence_predicted = sentence_predicted.replace(" ", "", 1)
        print(sentence_predicted)

        mapper_file = codecs.open("thumbnail/unicode_map.txt", "r", "utf-8-sig")
        maps = dict()
        lines = mapper_file.read().split("\n")
        mapper_file.close()

        for line in lines:
            key = line.split("\t")[0]
            value = line.split("\t")[1].replace("\r", "")
            maps[key] = value

        for key in maps:
            sentence_predicted = sentence_predicted.replace(key, maps.get(key))

        print(sentence_predicted)
        corrected = sc.spell_correct(sentence_predicted)
        print(corrected)
        return corrected
        # print(list(corrected))

    def predict_start(self):
        if self.files:
            keras.backend.clear_session()
            return self.identifyprocess()
        else:
            return "None"

# file_write = codecs.open("Text/" + img_no + ".txt", "w", "utf-16")
# file_write.write(sentence_predicted)

# sinhala_classes = []
# for key in mapped_classes:
#     sinhala_classes.append(maps[key])

# idx = [index for index, value in enumerate(mapped_classes) if value == 'ෙ']
# for i in idx:
#     mapped_classes[i], mapped_classes[i + 1] = mapped_classes[i + 1], mapped_classes[i]

# str_arr = [sinhala_classes[i: j] for i, j in zip([0] + indexes[1:], indexes[1:] + [None])]
# print(str_arr)
# out_str = ""
# file_write = codecs.open("Text/" + img_no + ".txt", "w", "utf-16")
#
# for st in str_arr:
#     for s in st:
#         out_str = out_str + s
#         file_write.write(s)
#     out_str = out_str + " "
#     file_write.write(" ")

# print(out_str)
# prdictions = []
# for st in str_arr:
#     prdictions.append(st)
#     prdictions.append(' ')
#
# sentence_predicted = ''.join(str(prdictions))
# print(sentence_predicted)


# mapper_file = codecs.open("unicode_map.txt", "r", "utf-8")
# maps = dict()
# lines = mapper_file.read().split("\n")
#
# for line in lines:
#     key = line.split("\t")[0]
#     value = line.split("\t")[1].replace("\r", "")
#     maps[key] = value
#     mapper_file.close()
#
# sinhala_classes = []
# for key in mapped_classes:
#     sinhala_classes.append(maps[key])
#
# print(sinhala_classes)
