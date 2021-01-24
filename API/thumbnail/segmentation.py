import os
import cv2
import numpy as np
import tensorflow as tf
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import skeletonize
from skimage.util import invert
from keras import backend as k
import operator


img_no = ''


def linesegment(binary, no):
    global img_no
    global model_segment
    model_segment = tf.keras.models.load_model('thumbnail/model_cnn_segment.pkl')
    global graph
    graph = tf.get_default_graph()
    img_no = no
    horizontalprofile = np.sum(binary, 1)

    # plt.plot(horizontalprofile)
    # plt.show()

    iszero = np.equal(horizontalprofile, 0).view(np.int8)
    # print(iszero)
    absdiff = np.abs(np.diff(iszero))
    # print(absdiff)
    row_gaps = np.where(absdiff == 1)[0].reshape(-1, 2)

    for i in range(len(row_gaps)):
        crop_line = binary[row_gaps[i][0]:row_gaps[i][1] + 1, 0:binary.shape[1]]
        # cv2.imshow('cropline_' + str(i+1), crop_line)
        wordsegment(crop_line, i + 1)
    k.clear_session()


def wordsegment(line, line_no):
    verticalprofile = np.sum(line, 0)
    # plt.plot(verticalprofile)
    # plt.show()

    # print(verticalprofile)
    # print('__________________________________________________________________________')
    iszero = np.equal(verticalprofile, 0).view(np.int8)
    # print(iszero)
    # print('__________________________________________________________________________')
    absdiff = np.abs(np.diff(iszero))
    # print(absdiff)
    # print('__________________________________________________________________________')
    try:
        col_gaps = np.where(absdiff == 1)[0].reshape(-1, 2)
    except:
        if absdiff[-1] == 1:
            absdiff[-1] = 0
        else:
            absdiff[-1] = 1
        col_gaps = np.where(absdiff == 1)[0].reshape(-1, 2)
    # print(absdiff)
    # print(col_gaps)
    gap = []
    for i in range(len(col_gaps) - 1):
        gap.append(abs(col_gaps[i][1] - col_gaps[i + 1][0]))
    # print(gap)
    gap.sort()
    # print(gap)
    diff = []
    for i in range(len(gap)-1):
        diff.append(gap[i+1]-gap[i])
    if not diff:
        diff.append(0)
    trun_diff = [i for i in diff if i < 20]
    if not trun_diff:
        trun_diff = diff
    index, value = max(enumerate(trun_diff), key=operator.itemgetter(1))
    # print(index)
    thr_index = index if len(gap) == 1 else index+1
    for i in range(len(col_gaps) - 1):
        if abs(col_gaps[i][1] - col_gaps[i + 1][0]) < max(gap[thr_index], 4):
            absdiff[col_gaps[i][1]] = 0
            absdiff[col_gaps[i + 1][0]] = 0

    col_gaps = np.where(absdiff == 1)[0].reshape(-1, 2)
    # print(absdiff)
    # print(col_gaps)
    # plt.show()
    for i in range(len(col_gaps)):
        crop_word = line[0:line.shape[0], col_gaps[i][0]:col_gaps[i][1] + 1]
        # cv2.imshow('crop_' + str(line_no) + '_' + str(i+1), crop_word)
        charsegment(crop_word, i + 1, line_no)


def charsegment(word, word_no, line_no):
    verticalp = np.sum(word, 0)
    # plt.plot(verticalp)
    iszero = np.equal(verticalp, 0).view(np.int8)
    # print(iszero)
    absdiff = np.abs(np.diff(iszero))
    # print(absdiff)
    # plt.show()
    try:
        absdiff[-1] = 1
        gaps = np.where(absdiff == 1)[0].reshape(-1, 2)
    except:
        absdiff[-1] = 0
        gaps = np.where(absdiff == 1)[0].reshape(-1, 2)
    # print(gaps)
    k = 0

    for i in range(len(gaps)):
        crop_char = word[0:word.shape[0], gaps[i][0]:gaps[i][1] + 2]
        crop_char = cv2.rotate(crop_char, cv2.ROTATE_90_CLOCKWISE)
        ret, labels = cv2.connectedComponents(crop_char)
        # print(labels)
        props = regionprops(labels)
        for prop in props:
            # print(prop['label'])
            cropped_shape = prop['image']
            # print(cropped_shape)
            cropped_shape = 255 * cropped_shape
            # print(cropped_shape)
            labeled_img = cv2.rotate(cropped_shape, cv2.ROTATE_90_COUNTERCLOCKWISE)

            labeled_img = np.array(labeled_img, dtype='uint8')
            image = rescale(labeled_img)
            image = invert(image)
            images = []
            image_copy = np.reshape(image, (50, 50, 1))
            images.append(image_copy)
            x = np.array(images)
            with graph.as_default():
                pre_class = model_segment.predict_classes(x)
            if pre_class[0] == 1:
                k = seg_connected(image, line_no, word_no, k)
            else:
                cv2.imwrite(os.path.join(img_no,
                                         'cropchar_' + str(line_no) + '_' + str(word_no) + '_' + str(k + 1) + '.png'),
                            image)
                k = k + 1
            # j = j+1
            # if j > 1:
            #     m = 1
            # else:
            #     m = 0

        # label_hue = np.uint8(255 * labels / np.max(labels))
        # print(label_hue)
        # blank_ch = 255 * np.ones_like(label_hue)
        # labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # cvt to BGR for display
        # labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # set bg label to black
        # labeled_img[label_hue == 0] = 0
        # labeled_img = cv2.rotate(labeled_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # print('****************************')
        # print(labels)
        # crop_char.thumbnail((128, 128), Image.ANTIALIAS)
        # resized = cv2.resize(crop_char, (30, 30), interpolation=cv2.INTER_AREA)
        # cv2.imshow('cropchar_' + str(line_no) + '_' + str(word_no) + '_' + str(i+1), out)
        # cv2.imwrite(os.path.join('F:/FYP/Development/Interim/text-detection/' + img_no,
        #                          'cropchar_' + str(line_no) + '_' + str(word_no) + '_' + str(i + 1) + '.png'),
        #             labeled_img)
    # cv2.waitKey()


def rescale(labeled_img):
    height, width = labeled_img.shape
    resize_height, resize_width = 50, 50
    left, right, top, bottom = 0, 0, 0, 0

    if width > resize_width and width >= height:
        wpercent = (resize_width / float(width))
        hsize = int((float(height) * float(wpercent)))
        hsize = hsize if hsize > 1 else 1
        labeled_img = cv2.resize(labeled_img, (resize_width, hsize))
        height, width = labeled_img.shape

    if height > resize_height and height >= width:
        hpercent = (resize_height / float(height))
        wsize = int((float(width) * float(hpercent)))
        wsize = wsize if wsize > 1 else 1
        labeled_img = cv2.resize(labeled_img, (wsize, resize_height))
        height, width = labeled_img.shape

    if width < resize_width:
        difference = resize_width - width

        left = difference // 2
        right = difference // 2

        if difference % 2 == 1:
            right = right + 1

    if height < resize_height:
        difference = resize_height - height

        top = difference // 2
        bottom = difference // 2

        if difference % 2 == 1:
            bottom = bottom + 1

    return cv2.copyMakeBorder(labeled_img, top=top, bottom=bottom, left=left, right=right,
                              borderType=cv2.BORDER_CONSTANT,
                              value=[0, 0, 0])


def seg_connected(org, line_no, word_no, k):
    image = cv2.copyMakeBorder(org, top=1, bottom=1, left=1, right=1,
                               borderType=cv2.BORDER_CONSTANT,
                               value=[255, 255, 255])
    img_new = scale(image, 500)
    skele = skeletonize(image // 255)
    skele = (skele * 255).astype(np.uint8)

    labels = label(skele)
    assert (labels.max() != 0)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    largestCC = (largestCC * 255).astype(np.uint8)

    large = scale(largestCC, 500)

    _, bw = cv2.threshold(large, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contour, hi = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = cv2.boundingRect(contour[0])

    kernel_h = int(h * 0.23)
    kernel = np.ones((kernel_h, 1), np.uint8)
    d_im = cv2.dilate(bw, kernel, iterations=1)
    e_im = cv2.erode(d_im, kernel, iterations=1)

    skeleton = skeletonize(e_im // 255)
    skeleton = (skeleton * 255).astype(np.uint8)

    contours, hierarchy = cv2.findContours(skeleton, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    idx = 0
    #
    for cnt in contours[1:]:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        mask = np.zeros(img_new.shape, dtype=np.uint8)
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        inv = invert(mask)
        out = np.zeros_like(img_new)
        out[mask == 255] = img_new[mask == 255]
        output = out + inv
        roi = output[y:y + h, x:x + w]
        resized_img = invert(rescale(invert(roi)))
        _, corrected_img = cv2.threshold(resized_img, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imwrite(os.path.join(img_no,
                                 'cropchar_' + str(line_no) + '_' + str(word_no) + '_' + str(k + 1) + '.png'),
                    corrected_img)
        k = k + 1
    return k


def scale(im, rh):
    height, width = im.shape
    hpercent = (rh / float(height))
    wsize = int((float(width) * float(hpercent)))
    return cv2.resize(im, (wsize, rh))
