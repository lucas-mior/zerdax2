from types import SimpleNamespace
import sys
import cv2
import copy
import numpy as np
import logging as log

import algorithm as algo
import drawings as draw
import yolov5.detect as yolo
import constants as consts
from misc import SYMBOLS, AMOUNT, NUMBERS

conf = consts.conf_thres
iou = consts.iou_thres


def detect_objects(img):
    objs = yolo.run(weights="best.pt",
                    source=img.filename,
                    data="zerdax2.yaml",
                    nosave=True,  # do not save images/videos
                    conf_thres=conf,  # confidence threshold
                    iou_thres=iou,  # NMS IOU threshold
                    max_det=32,  # maximum detections per image
                    save_txt=False,  # save results to *.txt
                    save_conf=True,  # save confidences in --save-txt labels
                    project='.',  # save results to project/name
                    name='exp',  # save results to project/name
                    exist_ok=True,  # existing project/name ok, don't increment
                    )

    objs = objs[np.argsort(objs[:, 4])][::-1]
    boardnum = NUMBERS['Board']
    for obj in objs:
        if obj[5] == boardnum:
            img.boardbox = np.array(obj[:4], dtype='int32')
            break

    log.info(f"{img.boardbox=}")

    img.pieces = objs[objs[:, 5] != boardnum].tolist()
    img.pieces = determine_colors(img.pieces, img.BGR)
    img.pieces = process_pieces(img.pieces)

    if algo.debugging():
        canvas = draw.boxes(img.pieces, img.BGR)
        draw.save("yolo", canvas)
    return img


def determine_colors(pieces, image):
    pcolors = []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for p in pieces:
        x0, y0 = int(p[0])+4, int(p[1])+4
        x1, y1 = int(p[2])-4, int(p[3])-7
        a = image[y0:y1, x0:x1]
        avg = np.median(a, overwrite_input=True)
        p.append(avg)
        pcolors.append(p)

    pcolors = np.array(pcolors, dtype='float32')

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compact, labels, centers = cv2.kmeans(pcolors[:, 6], 2, None,
                                          criteria, 20, flags)
    blacklabel = int(centers[0] > centers[1])

    for i, p in enumerate(pcolors):
        if labels[i] == blacklabel:
            p[5] += 6

    return pcolors.tolist()


def process_pieces(pieces):
    new_pieces = []
    rules = copy.deepcopy(AMOUNT)

    for piece in pieces:
        x0, y0, x1, y1, conf, num, _ = piece
        num = int(num)
        rule = rules[SYMBOLS[num]]
        if rule[0] < rule[1]:
            rule[0] += 1
            new_pieces.append(piece)

    return new_pieces


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        img = SimpleNamespace(filename=filename)
        img.BGR = cv2.imread(filename)
        img = detect_objects(img)
        canvas = draw.boxes(img.pieces, img.BGR)
        draw.save("yolo", canvas)
