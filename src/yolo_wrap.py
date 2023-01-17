import cv2
import copy
import numpy as np
import logging as log

import auxiliar as aux
import drawings as draw
import yolov5.detect as yolo
from zerdax2_misc import SYMBOLS, AMOUNT, NUMBERS


def detect_objects(img):
    objs = yolo.run(weights="best.pt",
                    source=img.filename,
                    data="yolov5/zerdax2.yaml",
                    nosave=True,  # do not save images/videos
                    conf_thres=0.5,  # confidence threshold
                    iou_thres=0.45,  # NMS IOU threshold
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

    canvas = draw.boxes(img.pieces, img.BGR)
    # aux.save(img, "yolo", canvas)
    return img


def determine_colors(pieces, image):
    pcolors = []
    for p in pieces:
        s = avg = weight = 0
        x0, y0 = int(p[0]), int(p[1])
        x1, y1 = int(p[2]), int(p[3])
        a = image[y0:y1, x0:x1]
        b = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        xc = round(b.shape[1]/2)
        yc = round(b.shape[0]/2)
        for (x, y), pixel in np.ndenumerate(b):
            r = np.sqrt((xc-x)**2 + (yc-y)**2)
            w = 1/(r + 1)
            weight += w
            s += pixel * w

        avg = round(s/weight, 2)
        p.append(avg)
        pcolors.append(p)

    pcolors = np.array(pcolors, dtype='float32')

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compact, labels, centers = cv2.kmeans(pcolors[:, 6], 2, None,
                                          criteria, 10, flags)
    if centers[0] < centers[1]:
        blacklabel = 0
    else:
        blacklabel = 1

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
