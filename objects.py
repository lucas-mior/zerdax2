import sys
import cv2
import copy
import numpy as np
from jenkspy import jenks_breaks
from ultralytics import YOLO

import draw
from misc import SYMBOLS, AMOUNT, NUMBERS
import algorithm


def detect(BGR):
    model = YOLO("zerdax2.pt")
    objects = model.predict(source=BGR,
                            conf=0.5,
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
    hsvalue = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_and(hsvalue, gray)

    avg_colors = np.empty(len(pieces), dtype='int32')
    for i, p in enumerate(pieces):
        x0, y0, x1, y1 = p[:4]
        x0 += 8
        y0 += 8
        x1 -= 2
        y1 -= 10
        box = image[y0:y1, x0:x1]
        avg_colors[i] = np.median(box, overwrite_input=True)

    try:
        limits = jenks_breaks(avg_colors, n_classes=2)
        black = pieces[avg_colors <= limits[1]]
        white = pieces[avg_colors > limits[1]]
        black[:, 5] += 6
        pieces = np.vstack((black, white))
    except Exception:
        pass
    return pieces


def remove_captured_pieces(pieces, boardbox):
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
        BGR = cv2.imread(filename)

        boardbox, pieces = detect(BGR)
        canvas = draw.boxes(BGR, pieces, boardbox)
        draw.save("", canvas, title="0detection.png")

        pieces = determine_colors(pieces, BGR)
        canvas = draw.boxes(BGR, pieces, boardbox)
        draw.save("", canvas, title="1colors.png")

        pieces = remove_captured_pieces(pieces, boardbox)
        canvas = draw.boxes(BGR, pieces, boardbox)
        draw.save("", canvas, title="2remove_captured.png")

        pieces = process_pieces_amount(pieces)
        canvas = draw.boxes(BGR, pieces, boardbox)
        draw.save("", canvas, title="3amount_fix.png")
