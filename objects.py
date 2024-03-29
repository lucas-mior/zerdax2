import sys
import cv2
import copy
import numpy as np
from ultralytics import YOLO

import draw
from misc import SYMBOLS, AMOUNT, NUMBERS
import algorithm


def detect(BGR):
    model = YOLO("zerdax2.pt")
    objects = model.predict(source=BGR,
                            conf=0.4,
                            # device="cpu",
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
    if boardbox is None or algorithm.debug:
        canvas = draw.boxes(BGR, pieces, boardbox)
        draw.save("detection", canvas)
    return boardbox, pieces


def determine_colors(pieces, image):
    if len(pieces) == 0:
        return pieces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hue = hsv[:, :, 0]
    hue = cv2.medianBlur(hue, 3)
    hue = cv2.equalizeHist(hue)

    avg_colors = np.empty((len(pieces), 2), dtype='float32')

    def value_map(value, in_min, in_max, out_min, out_max):
        proportion = (value - in_min) / (in_max - in_min)
        mapped_value = (proportion * (out_max - out_min)) + out_min
        return mapped_value

    for i, p in enumerate(pieces):
        x0, y0, x1, y1 = p[:4]
        dx = x1 - x0
        dy = y1 - y0

        if dx/dy > 0.6 and dx > 35:
            box = gray[y0:y1, x0:x1]
            boxh = hue[y0:y1, x0:x1]
            mask = 255*np.ones(boxh.shape, dtype='uint8')
            a = dy/(dx/2)
            if x0 < gray.shape[1]/2:
                for (y, x), pixel in np.ndenumerate(mask):
                    if x < dx/2 and y > x*a:
                        mask[y, x] = 0
                    if x > dx/2 and (dy - y) > (dx - x)*a:
                        mask[y, x] = 0
            else:
                for (y, x), pixel in np.ndenumerate(mask):
                    if x < dx/2 and (dy - y) > x*a:
                        mask[y, x] = 0
                    if x > dx/2 and y > (dx - x)*a:
                        mask[y, x] = 0
        else:
            x0 += 5
            x1 -= 5
            y0 += 5
            y1 -= 3
            box = gray[y0:y1, x0:x1]
            boxh = hue[y0:y1, x0:x1]
            mask = 255*np.ones(boxh.shape, dtype='uint8')

        avg_colors[i, 0] = np.median(box[mask != 0])
        avg_colors[i, 1] = np.median(boxh[mask != 0])

    a0 = max(1, np.std(avg_colors[:, 0]))
    a1 = max(1, np.std(avg_colors[:, 1]))
    avg_colors[:, 1] *= (a0/a1)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, labels0, centers = cv2.kmeans(avg_colors, 2, None,
                                       criteria, 30, cv2.KMEANS_RANDOM_CENTERS)
    labels0 = np.ravel(labels0)
    black = pieces[(labels0 == 0)]
    white = pieces[(labels0 == 1)]
    if centers[1, 0] < centers[0, 0]:
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
        symbol = SYMBOLS[int(num)]
        rule = rules[symbol]
        if rule[0] < rule[1]:
            rule[0] += 1
            new_pieces.append(piece)
        elif symbol == 'K':
            rule = rules['Q']
            if rule[0] < rule[1]:
                rule[0] += 1
                piece[5] = NUMBERS['Q']
                new_pieces.append(piece)
        elif symbol == 'Q':
            rule = rules['K']
            if rule[0] < rule[1]:
                rule[0] += 1
                piece[5] = NUMBERS['K']
                new_pieces.append(piece)
        elif symbol == 'k':
            rule = rules['q']
            if rule[0] < rule[1]:
                rule[0] += 1
                piece[5] = NUMBERS['q']
                new_pieces.append(piece)
        elif symbol == 'q':
            rule = rules['k']
            if rule[0] < rule[1]:
                rule[0] += 1
                piece[5] = NUMBERS['k']
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
