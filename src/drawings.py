import numpy as np
import cv2
from zerdax2_misc import COLORS, SYMBOLS


def intersections(image, inter):
    canvas = np.zeros(image.shape, dtype='uint8')

    for i, p in enumerate(inter):
        cv2.circle(canvas, p, radius=5,
                   color=(30+i*2, 0, 225-i*2), thickness=-1)

    cv2.addWeighted(image, 0.5, canvas, 0.5, 0, canvas)
    return canvas


def lines(image, vert, hori=None):
    canvas = np.zeros(image.shape, dtype='uint8')

    if hori is not None:
        for x1, y1, x2, y2, r, t in vert:
            cv2.line(canvas, (x1, y1), (x2, y2),
                     color=(255, 0, 0), thickness=3)  # blue
        for x1, y1, x2, y2, r, t in hori:
            cv2.line(canvas, (x1, y1), (x2, y2),
                     color=(0, 255, 0), thickness=3)  # green
    else:
        for x1, y1, x2, y2, r, t in vert:
            cv2.line(canvas, (x1, y1), (x2, y2),
                     color=(0, 0, 255), thickness=3)  # red

    cv2.addWeighted(image, 0.5, canvas, 0.5, 0, canvas)
    return canvas


def corners(image, BR, BL, TR, TL):
    canvas = np.zeros(image.shape, dtype='uint8')

    cv2.circle(canvas, BR, radius=7,
               color=(255, 0, 0), thickness=-1)
    cv2.circle(canvas, BL, radius=7,
               color=(0, 255, 0), thickness=-1)
    cv2.circle(canvas, TR, radius=7,
               color=(0, 0, 255), thickness=-1)
    cv2.circle(canvas, TL, radius=7,
               color=(255, 255, 0), thickness=-1)

    cv2.addWeighted(image, 0.5, canvas, 0.5, 0, canvas)
    return canvas


def squares(image, squares):
    canvas = np.zeros(image.shape, dtype='uint8')
    scale = 2.5 * (image.shape[1]/1920)

    def _draw_square(canvas, coord, color, name):
        cv2.drawContours(canvas, [coord], -1, color=color, thickness=3)
        cv2.putText(canvas, name, (coord[0, 0]+5, coord[0, 1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color=color, thickness=2)
        return canvas

    canvas = _draw_square(canvas, squares[0, 0], (255, 0, 0), "A1")
    canvas = _draw_square(canvas, squares[4, 3], (0, 255, 0), "E4")
    canvas = _draw_square(canvas, squares[2, 4], (0, 0, 255), "C5")
    canvas = _draw_square(canvas, squares[7, 7], (0, 255, 255), "H8")

    cv2.addWeighted(image, 0.5, canvas, 0.5, 0, canvas)
    return canvas


def boxes(pieces, image):
    canvas = np.zeros(image.shape, dtype='uint8')
    thick = round(2.4 * (image.shape[0] / 1280))

    for piece in pieces:
        x0, y0, x1, y1, conf, num, _ = piece
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        conf, num = round(float(conf), 2), int(num)
        color = COLORS[num]
        symbol = SYMBOLS[num]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color=color, thickness=thick)
        cv2.putText(canvas, f"{symbol} {conf}", (x0-5, y0-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, thick)

    cv2.addWeighted(image, 0.5, canvas, 0.5, 0, canvas)
    return canvas
