import sys
import numpy as np
import algorithm as algo
import cv2
import logging as log

from misc import COLORS, SYMBOLS
import lines as li


def save(name, image, title=None):
    if not hasattr(save, "i"):
        save.i = 0
    save.i += 1
    if title is None:
        title = f"{algo.img.basename}_{save.i:04d}_{name}.png"
    log.info(f"saving {title}...")
    cv2.imwrite(title, image)
    return


def addweighted(image, canvas, w1=0.4, w2=0.6):
    cv2.addWeighted(image, w1, canvas, w2, 0, canvas)
    return canvas


def corners(image, corners):
    canvas = np.zeros(image.shape, dtype='uint8')

    for i, c in enumerate(corners):
        cv2.circle(canvas, c, radius=5,
                   color=(20+i*40, 0, 100+i*30), thickness=-1)

    canvas = addweighted(image, canvas)
    return canvas


def points(image, inters):
    canvas = np.zeros(image.shape, dtype='uint8')

    for i, row in enumerate(inters):
        for j, p in enumerate(row):
            cv2.circle(canvas, p, radius=5,
                       color=(20+i*20, 0, 100+j*15), thickness=-1)

    canvas = addweighted(image, canvas)
    return canvas


def lines(image, vert, hori=None, annotate_number=False):
    canvas = np.zeros(image.shape, dtype='uint8')

    def _draw(canvas, lines, color, kind=-1, annotate_number=True):
        for i, line in enumerate(lines[:, :4]):
            x0, y0, x1, y1 = line
            theta = round(li.theta(line))
            cv2.line(canvas, (x0, y0), (x1, y1),
                     color=color, thickness=2)
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

                cv2.putText(canvas, f"{i}.{theta}", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                            color=color, thickness=2)
        return canvas

    if hori is not None:
        hori = np.array(hori)
        canvas = _draw(canvas, hori, (0, 255, 0), 1, annotate_number)
        if vert is not None:
            vert = np.array(vert)
            canvas = _draw(canvas, vert, (255, 0, 80), 0, annotate_number)
        canvas = addweighted(image, canvas)
    elif vert is not None:
        vert = np.array(vert)
        canvas = _draw(canvas, vert, (0, 0, 255), annotate_number=False)
        canvas = addweighted(image, canvas)

    return canvas


def squares(image, squares):
    canvas = np.zeros(image.shape, dtype='uint8')
    scale = 2.5 * (image.shape[1]/1920)

    def _draw_square(canvas, coord, color, name):
        cv2.drawContours(canvas, [coord], -1, color=color, thickness=3)
        color = [max(0, c - 80) for c in color]
        cv2.putText(canvas, name, (coord[0, 0]+5, coord[0, 1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color=color, thickness=2)
        return canvas

    canvas = _draw_square(canvas, squares[0, 0, :4], [255, 0, 0], "A1")
    canvas = _draw_square(canvas, squares[4, 3, :4], [0, 255, 0], "E4")
    canvas = _draw_square(canvas, squares[2, 4, :4], [0, 0, 255], "C5")
    canvas = _draw_square(canvas, squares[7, 7, :4], [0, 255, 255], "H8")

    canvas = addweighted(image, canvas)
    return canvas


def boxes(image, pieces):
    canvas = np.zeros(image.shape, dtype='uint8')
    thick = round(2.4 * (image.shape[0] / 1280))

    for piece in pieces:
        x0, y0, x1, y1, conf, num = piece[:6]
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        conf, num = round(float(conf), 2), int(num)
        color = COLORS[num]
        symbol = SYMBOLS[num]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color=color, thickness=thick)
        cv2.putText(canvas, f"{symbol} {conf}", (x0-5, y0-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, thick)

    canvas = addweighted(image, canvas)
    return canvas


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        image = cv2.imread(sys.argv[1])
        canvas = cv2.imread(sys.argv[2])
        canvas = addweighted(image, canvas)
        save("addweighted", canvas)
