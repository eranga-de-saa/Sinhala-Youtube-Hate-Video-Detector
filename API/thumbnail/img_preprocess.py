import os
import cv2
import keras
import numpy as np
import thumbnail.segmentation as seg
import thumbnail.predict as predict


def process():
    # read image
    rgb = cv2.imread('temp/img.jpg')
    # cv2.imshow('original', rgb)

    # convert image to grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    # bilateral filter
    blur = cv2.bilateralFilter(gray, 5, 75, 75)
    # cv2.imshow('blur', blur)

    # morphological gradient calculation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
    # cv2.imshow('gradient', grad)

    # binarization
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow('otsu', bw)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (32, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 32))
    detected_hrlines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horizontal_kernel)
    detected_vrlines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vertical_kernel)
    # cv2.imshow('lih', detected_hrlines)
    # cv2.imshow('liv', detected_vrlines)

    bw = bw - detected_hrlines - detected_vrlines
    # cv2.imshow('sdad', bw)

    # closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 1))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('closed', closed)

    black_bg_copy = np.zeros_like(gray)

    # finding contours
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(closed.shape, dtype=np.uint8)

    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y + h, x:x + w] = 0
        area = cv2.contourArea(contours[idx])
        aspect_ratio = float(w) / h
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)

        # identify region of interest
        if r > 0.34 and 0.52 < aspect_ratio < 13 and area > 145.0:
            selected = gray.copy()[y:y + h, x:x + w]
            _, bw_final = cv2.threshold(selected, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # cv2.rectangle(rgb, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
            # print("top" + str(bw_final[0, 0]))
            # print("bot" + str(bw_final[h - 1, w - 1]))
            # cv2.imshow('black', bw_final)
            if bw_final[0, 0] == 0 or bw_final[h - 1, w - 1] == 0:
                black_bg_copy[y:y + h, x:x + w] = bw_final
            else:
                inverted = cv2.bitwise_not(bw_final)
                black_bg_copy[y:y + h, x:x + w] = inverted

    # cv2.imshow('rectangles', rgb)
    # cv2.imshow('result', black_bg)
    # cv2.imshow('output', black_bg_copy)

    cnts, high = cv2.findContours(black_bg_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_mask = np.zeros(black_bg_copy.shape, dtype=np.uint8)
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 10:
            cv2.drawContours(new_mask, [c], -1, (255, 255, 255), -1)

    result = cv2.bitwise_and(black_bg_copy, new_mask)
    # cv2.imshow('result', result)

    try:
        # Create target Directory
        os.mkdir('img_folder')
        print("Directory ", 'img_folder', " Created ")
    except FileExistsError:
        print("Directory ", 'img_folder', " already exists")

    # black_bg_copy[black_bg_copy == 255] = 1
    # inverted = invert(black_bg_copy)
    # skeleton = skeletonize(inverted//255)
    # ske = (skeleton * 255).astype(np.uint8)
    # cv2.imshow("sk", ske)

    # black_bg_copy[black_bg_copy == 255] = 1
    # skeleton = skeletonize(black_bg_copy)
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
    #                          sharex=True, sharey=True)
    #
    # ax = axes.ravel()
    #
    # ax[1].imshow(skeleton, cmap=plt.cm.gray)
    # ax[1].set_title('skeletonize')
    # ax[1].axis('off')
    #
    # fig.tight_layout()
    # plt.show()
    # size = np.size(black_bg_copy)
    # skel = np.zeros(black_bg_copy.shape, np.uint8)
    #
    # ret, img = cv2.threshold(black_bg_copy, 127, 255, 0)
    # element = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))
    # done = False
    #
    # while (not done):
    #     eroded = cv2.erode(img, element)
    #     temp = cv2.dilate(eroded, element)
    #     temp = cv2.subtract(img, temp)
    #     skel = cv2.bitwise_or(skel, temp)
    #     img = eroded.copy()
    #
    #     zeros = size - cv2.countNonZero(img)
    #     if zeros == size:
    #         done = True
    #
    # cv2.imshow("skel", ske)

    seg.linesegment(result, 'img_folder')

    # avg_width = wid_seg.linesegment(result)
    # wid_seg_save.linesegment(result, img_no, avg_width)
    print('img_folder' + " Done")
    keras.backend.clear_session()
    p = predict.prediction('img_folder')
    return p.predict_start()

    # cv2.waitKey()
