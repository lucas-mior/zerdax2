import sys
import numpy as np
import cv2
import logging as log

import algorithm

from misc import COLORS, SYMBOLS
import lines as li

WIDTH_INPUT = 960


def adapt(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) != 3:
        log.error(f"image.shape: {image.shape}")
        exit(1)
    return image


def save(name, image, title=None):
    if not hasattr(save, "i"):
        save.i = 0
    save.i += 1
    if title is None:
        title = f"{algorithm.basename}_{save.i:04d}_{name}.png"
    log.debug(f"saving {title}...")
    cv2.imwrite(title, image)
    return


def add_weighted(image, canvas, w1=0.4, w2=0.6):
    image = adapt(image)
    cv2.addWeighted(image, w1, canvas, w2, 0, canvas)
    return canvas


def corners(image, corners):
    image = adapt(image)
    min_x = np.min(corners[:, 0])
    min_y = np.min(corners[:, 1])
    max_x = np.max(corners[:, 0])
    max_y = np.max(corners[:, 1])

    pad_left = max(0, -min_x) + 4
    pad_right = max(0, -(image.shape[1] - max_x)) + 4
    pad_up = max(0, -min_y) + 4
    pad_down = max(0, -(image.shape[0] - max_y)) + 4

    image = cv2.copyMakeBorder(image, pad_up, pad_down, pad_left, pad_right,
                               cv2.BORDER_CONSTANT)
    corners = np.copy(corners)
    corners[:, 0] += pad_left
    corners[:, 1] += pad_up

    canvas = np.zeros(image.shape, dtype='uint8')
    radius = round(5 * (image.shape[1] / 512))

    name = ["TL", "TR", "BR", "BL"]
    thick = round(2 * (image.shape[1] / 512))
    for i, c in enumerate(corners):
        color = (i*50, 30*i, 255-i*40)
        cv2.circle(canvas, c, radius,
                   color=color, thickness=-1)
        cv2.putText(canvas, name[i], c,
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                    color=color, thickness=thick)

    canvas = add_weighted(image, canvas)
    return canvas


def points(image, inters):
    image = adapt(image)
    canvas = np.zeros(image.shape, dtype='uint8')
    radius = round(5 * (image.shape[1] / 512))

    for i, row in enumerate(inters):
        for j, p in enumerate(row):
            cv2.circle(canvas, p, radius,
                       color=(20+i*20, 0, 100+j*15), thickness=-1)

    canvas = add_weighted(image, canvas)
    return canvas


def lines(image, vert, hori=None, annotate_number=False):
    image = adapt(image)
    canvas = np.zeros(image.shape, dtype='uint8')
    thick = round(2 * (image.shape[1] / 512))

    def _draw(canvas, lines, color, kind=-1):
        if kind != -1:
            ll = len(lines)
            legend = f"{ll} vertical" if kind == 0 else f"{ll} horizontal"
            cv2.putText(canvas, legend, (20, 20+30*kind),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                        color=color, thickness=thick)
        else:
            ll = len(lines)
            legend = f"{ll} lines"
            cv2.putText(canvas, legend, (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                        color=color, thickness=thick)
        for i, line in enumerate(lines[:, :4]):
            x0, y0, x1, y1 = line
            angle = round(li.theta(line))
            cv2.line(canvas, (x0, y0), (x1, y1),
                     color=color, thickness=thick)
            if annotate_number:
                x, y = x0 + 10, y0 + 10
                if kind == 0:
                    x += 15*i
                    y += 30*i
                    if i == (len(lines)-1):
                        y -= 100
                        x -= 30
                else:
                    x += 5*i
                    if i == (len(lines)-1):
                        y -= 20
                        x -= 80

                cv2.putText(canvas, f"{i}.{angle}", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                            color=color, thickness=thick)
        return canvas

    if hori is not None:
        hori = np.array(hori)
        canvas = _draw(canvas, hori, (0, 255, 0), 1)
        if vert is not None:
            vert = np.array(vert)
            canvas = _draw(canvas, vert, (255, 0, 80), 0)
        canvas = add_weighted(image, canvas)
    elif vert is not None:
        vert = np.array(vert)
        canvas = _draw(canvas, vert, (0, 0, 255))
        canvas = add_weighted(image, canvas)

    return canvas


def squares(image, squares):
    image = adapt(image)
    canvas = np.zeros(image.shape, dtype='uint8')
    scale = round(1 + round(image.shape[1] / 1024))

    def _get_coord(name):
        letter = name[0].upper()
        row = int(name[1]) - 1
        col = ord(letter) - ord('A')
        return col, row

    def _draw_square(canvas, name):
        col, row = _get_coord(name)
        coord = squares[col, row, :4]
        color = COLORS[col]
        colortext = [max(0, c - 50) for c in color]

        cv2.drawContours(canvas, [coord], -1, color=color, thickness=3)
        cv2.putText(canvas, name, (coord[0, 0]+5, coord[0, 1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, colortext, thickness=2)
        return canvas

    for name in ["A1", "B2", "C3", "D4", "E5", "F6", "G7", "H8"]:
        canvas = _draw_square(canvas, name)

    canvas = add_weighted(image, canvas)
    return canvas


def boxes(image, pieces, boardbox=None):
    image = adapt(image)
    canvas = np.zeros(image.shape, dtype='uint8')
    thick = round(2 * (max(image.shape[1], image.shape[0]) / 1280))

    for piece in pieces:
        x0, y0, x1, y1, conf, num = piece[:6]
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        conf, num = round(conf/1000, 2), int(num)
        color = COLORS[num]
        symbol = SYMBOLS[num]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color=color, thickness=thick)
        cv2.putText(canvas, f"{symbol} {conf}", (x0-5, y0-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, thick)
    if boardbox is not None:
        x0, y0, x1, y1 = boardbox[:4]
        color = COLORS[0]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color=color, thickness=thick)
        cv2.putText(canvas, "Board", (x0-5, y0-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, thick)

    canvas = add_weighted(image, canvas)
    return canvas


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        image = cv2.imread(sys.argv[1])
        canvas = cv2.imread(sys.argv[2])
        canvas = add_weighted(image, canvas)
        save("add_weighted", canvas)
