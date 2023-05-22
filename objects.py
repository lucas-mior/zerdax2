import sys
import cv2
import copy
import numpy as np
from jenkspy import jenks_breaks
from ultralytics import YOLO

import draw
from misc import SYMBOLS, AMOUNT, NUMBERS


def detect(filename):
    model = YOLO("zerdax2.pt")
    objects = model.predict(source=filename,
                            conf=0.5,
                            device="cpu",
                            imgsz=640,
                            iou=0.7,
                            max_det=32)

    objects = objects[0].boxes
    confidences = np.array(objects.conf.cpu())
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
        confidence = piece.conf[0].cpu()
        klass = piece.cls[0].cpu()
        npieces.append([x0, y0, x1, y1, confidence, klass])

    pieces = np.array(npieces, dtype='O')
    pieces[:, :4] = np.int32(pieces[:, :4])
    pieces[:, 4] = np.float64(pieces[:, 4])
    pieces[:, 5] = np.int32(pieces[:, 5])

    return boardbox, pieces


def determine_colors(pieces, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    avg_colors = np.empty(len(pieces), dtype='int32')
    for i, p in enumerate(pieces):
        x0, y0 = p[0] + 4, p[1] + 4
        x1, y1 = p[2] - 4, p[3] - 7
        box = image[y0:y1, x0:x1]
        avg_colors[i] = np.median(box, overwrite_input=True)

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
        BGR = cv2.imread(filename)
        boardbox, pieces = detect(filename)
        canvas = draw.boxes(BGR, pieces, boardbox)
        draw.save("detection", canvas, title="detection.png")
