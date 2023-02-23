from types import SimpleNamespace
import sys
import cv2
import copy
import numpy as np
from jenkspy import jenks_breaks
from ultralytics import YOLO

import algorithm as algo
import drawings as draw
import constants as consts
from misc import SYMBOLS, AMOUNT, NUMBERS

conf = consts.conf_thres
iou = consts.iou_thres


def detect_objects(img):
    model = YOLO("zerdax2.pt")
    objs = model.predict(source=img.filename,
                         conf=0.25,
                         iou=0.7,
                         max_det=32)
    objs = objs[0].numpy().boxes

    objs = objs[np.argsort(objs.conf)][::-1]

    boardnum = NUMBERS['Board']
    for obj in objs:
        if obj.cls == boardnum:
            img.boardbox = np.array(obj.xyxy, dtype='int32')[0]
            break

    pieces = objs[objs.cls != boardnum]

    npieces = []
    for piece in pieces:
        x0, y0, x1, y1 = piece.xyxy[0]
        conf, cls = piece.conf[0], piece.cls[0]
        npieces.append([x0, y0, x1, y1, conf, cls])

    pieces = np.array(npieces, dtype='O')
    pieces[:, :4] = np.int32(pieces[:, :4])
    pieces[:, 5] = np.int32(pieces[:, 5])

    img.pieces = determine_colors(pieces, img.BGR)
    img.pieces = pieces
    img.pieces = img.pieces[np.argsort(img.pieces[4])]
    img.pieces = process_pieces(img.pieces)

    if algo.debug:
        canvas = draw.boxes(img.BGR, img.pieces)
        draw.save("yolo", canvas)
    return img


def determine_colors(pieces, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    avg_colors = np.empty(len(pieces), dtype='int32')
    for i, p in enumerate(pieces):
        x0, y0 = p[0] + 4, p[1] + 4
        x1, y1 = p[2] - 4, p[3] - 7
        a = image[y0:y1, x0:x1]
        avg_colors[i] = np.median(a, overwrite_input=True)

    limits = jenks_breaks(avg_colors, n_classes=2)

    black = pieces[avg_colors <= limits[1]]
    white = pieces[avg_colors > limits[1]]

    black[:, 5] += 6

    return np.vstack((black, white))


def process_pieces(pieces):
    new_pieces = []
    rules = copy.deepcopy(AMOUNT)

    for piece in pieces:
        x0, y0, x1, y1, conf, num = piece[:6]
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
        canvas = draw.boxes(img.BGR, img.pieces)
        draw.save("yolo", canvas, title="demo.png")
