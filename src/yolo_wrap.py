import cv2
import copy
import numpy as np

import auxiliar as aux
import drawings as draw
import yolov5.detect as yolo
from zerdax2_misc import SYMBOLS, AMOUNT


def process_pieces(img):
    new_pieces = []
    got = copy.deepcopy(AMOUNT)
    img.pieces = sorted(img.pieces, key=lambda x: x[4], reverse=True)

    for piece in img.pieces:
        x0, y0, x1, y1, conf, num, _ = piece
        num = str(int(num))
        if got[SYMBOLS[num]][0] < got[SYMBOLS[num]][1]:
            got[SYMBOLS[num]][0] += 1
            new_pieces.append(piece)

    img.pieces = new_pieces
    return img


def detect_objects(img):
    pieces = yolo.run(weights="best.pt",
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
                      )

    img.pieces = pieces.tolist()
    for obj in img.pieces:
        if obj[5] == 0:
            img.boardbox = obj
            img.pieces.remove(obj)
    print(f"{img.boardbox=}")
    print(f"{img.pieces=}")

    img = determine_colors(img)
    img = process_pieces(img)

    img = draw.boxes(img)
    aux.save(img, "yolo", img.yolopieces)
    return img


def determine_colors(img):
    pcolors = []
    for i, p in enumerate(img.pieces):
        avg = 0
        weight = 0
        x0, y0 = int(p[0]), int(p[1])
        x1, y1 = int(p[2]), int(p[3])
        xc = round((x1+x0)/2)
        yc = round((y1+y0)/2)
        a = img.BGR[y0:y1, x0:x1]
        b = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        for (x, y), pixel in np.ndenumerate(b):
            w = 1/max(abs(x-xc), 1) + 1/max(abs(y-yc), 1)
            weight += w
            avg += (pixel * w)
        avg = round(avg/weight, 2)
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

    img.pieces = pcolors.tolist()

    return img
