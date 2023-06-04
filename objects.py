import sys
import cv2
import copy
import numpy as np
from ultralytics import YOLO

import draw
from misc import SYMBOLS, AMOUNT, NUMBERS
import algorithm

WIDTH_INPUT = 960


def detect(BGR):
    model = YOLO("zerdax2.pt")
    objects = model.predict(source=BGR,
                            conf=0.7,
                            device="cpu",
                            imgsz=960,
                            iou=0.7,
                            max_det=33)

    objects = objects[0].boxes
    confidences = np.array(objects.conf.cpu())[::-1]
    objects = objects[np.argsort(confidences)]

    boardbox = None
    boardnum = NUMBERS['Board']
    for obj in objects:
        if obj.cls == boardnum:
            boardbox = np.array(obj.xyxy.cpu(), dtype='int32')[0]
            break

    pieces = objects[objects.cls != boardnum]

    npieces = []
    for piece in pieces:
        x0, y0, x1, y1 = piece.xyxy[0].cpu()
        confidence = np.round(piece.conf[0].cpu() * 1000)
        klass = piece.cls[0].cpu()
        npieces.append([x0, y0, x1, y1, confidence, klass])

    pieces = np.array(npieces, dtype='int32')
    if algorithm.debug:
        canvas = draw.boxes(BGR, pieces, boardbox)
        draw.save("detection", canvas)
    return boardbox, pieces


def determine_colors(pieces, image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    avg_colors = np.empty((len(pieces), 3), dtype='float32')
    for i, p in enumerate(pieces):
        x0, y0, x1, y1 = p[:4]
        x0 += 5
        y0 += 5
        x1 -= 5
        y1 -= 5
        box_h = hsv[y0:y1, x0:x1, 0]
        box_s = hsv[y0:y1, x0:x1, 1]
        box_v = hsv[y0:y1, x0:x1, 2]
        avg_colors[i, 0] = np.median(box_h)
        avg_colors[i, 1] = np.median(box_s)
        avg_colors[i, 2] = np.median(box_v)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, labels, centers = cv2.kmeans(avg_colors, 2, None,
                                      criteria, 30, cv2.KMEANS_RANDOM_CENTERS)
    labels = np.ravel(labels)
    print(centers)
    black = pieces[labels == 0]
    white = pieces[labels == 1]
    if centers[1, 2] < centers[0, 2]:
        aux = black
        black = white
        white = aux

    black[:, 5] += 6
    pieces = np.vstack((black, white))
    return pieces


def remove_captured_pieces(pieces, boardbox):
    if boardbox is None:
        return pieces

    xmin = np.minimum(pieces[:, 2], boardbox[2])
    xmax = np.maximum(pieces[:, 0], boardbox[0])
    inter_x = np.maximum(0, xmin - xmax)

    ymin = np.minimum(pieces[:, 3], boardbox[3])
    ymax = np.maximum(pieces[:, 1], boardbox[1])
    inter_y = np.maximum(0, ymin - ymax)

    inter_area = inter_x * inter_y

    dx = pieces[:, 2] - pieces[:, 0]
    dy = pieces[:, 3] - pieces[:, 1]
    boardbox_area = dx * dy

    area_ratio = inter_area / boardbox_area

    return pieces[area_ratio >= 0.25]


def process_pieces_amount(pieces):
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
        basename = str.rsplit(filename, ".", 1)[0]
        basename = str.rsplit(basename, "/", 1)[-1]

        BGR = cv2.imread(filename)

        boardbox, pieces = detect(BGR)
        canvas = draw.boxes(BGR, pieces, boardbox)
        draw.save("", canvas, title=f"{basename}_0detection.png")

        pieces = determine_colors(pieces, BGR)
        canvas = draw.boxes(BGR, pieces, boardbox)
        draw.save("", canvas, title=f"{basename}_1colors.png")

        # pieces = remove_captured_pieces(pieces, boardbox)
        # canvas = draw.boxes(BGR, pieces, boardbox)
        # draw.save("", canvas, title=f"{basename}_2remove_captured.png")

        # pieces = process_pieces_amount(pieces)
        # canvas = draw.boxes(BGR, pieces, boardbox)
        # draw.save("", canvas, title=f"{basename}_3amount_fix.png")
