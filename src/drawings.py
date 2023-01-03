import numpy as np
import cv2
from zerdax2_misc import COLORS, SYMBOLS


def intersections(image, inter):
    canvas = np.zeros(image.shape, dtype='uint8')
    for i, p in enumerate(inter):
        cv2.circle(canvas, p, radius=5,
                   color=(30+i*2, 0, 225-i*2), thickness=-1)
    cv2.addWeighted(image, 0.6, canvas, 0.4, 0, canvas)
    return canvas


def lines(img, image, vert, hori):
    canvas = np.zeros(image.shape, dtype='uint8')

    for x1, y1, x2, y2, r, t in vert:
        cv2.line(canvas, (x1, y1), (x2, y2),
                 color=(255, 0, 0), thickness=img.thick)
    for x1, y1, x2, y2, r, t in hori:
        cv2.line(canvas, (x1, y1), (x2, y2),
                 color=(0, 255, 0), thickness=img.thick)

    cv2.addWeighted(image, 0.6, canvas, 0.4, 0, canvas)
    return canvas


def corners(img, image, BR, BL, TR, TL):
    canvas = np.zeros(image.shape, dtype='uint8')
    cv2.circle(canvas, BR, radius=7,
               color=(255, 0, 0), thickness=-1)
    cv2.circle(canvas, BL, radius=7,
               color=(0, 255, 0), thickness=-1)
    cv2.circle(canvas, TR, radius=7,
               color=(0, 0, 255), thickness=-1)
    cv2.circle(canvas, TL, radius=7,
               color=(255, 255, 0), thickness=-1)

    cv2.addWeighted(image, 0.6, canvas, 0.4, 0, canvas)
    return canvas


def squares(img, image):
    canvas = np.zeros(image.shape, dtype='uint8')
    scale = 2.5 * (image.shape[1]/1920)
    cv2.drawContours(canvas, [img.sqback[0, 0]], -1,  # A1
                     color=(255, 0, 0), thickness=img.thick)
    cv2.putText(canvas, "A1", img.sqback[0, 0, 0]+5,
                cv2.FONT_HERSHEY_SIMPLEX, scale,
                color=(255, 0, 0), thickness=2)
    cv2.drawContours(canvas, [img.sqback[4, 3]], -1,  # E4
                     color=(0, 255, 0), thickness=img.thick)
    cv2.putText(canvas, "E4", img.sqback[4, 3, 0]+5,
                cv2.FONT_HERSHEY_SIMPLEX, scale,
                color=(0, 255, 0), thickness=2)
    cv2.drawContours(canvas, [img.sqback[2, 4]], -1,  # C5
                     color=(0, 0, 255), thickness=img.thick)
    cv2.putText(canvas, "C5", img.sqback[2, 4, 0]+5,
                cv2.FONT_HERSHEY_SIMPLEX, scale,
                color=(0, 0, 255), thickness=2)
    cv2.drawContours(canvas, [img.sqback[7, 7]], -1,  # H8
                     color=(0, 220, 220), thickness=img.thick)
    cv2.putText(canvas, "H8", img.sqback[7, 7, 0]+5,
                cv2.FONT_HERSHEY_SIMPLEX, scale,
                color=(0, 220, 220), thickness=2)

    cv2.addWeighted(image, 0.6, canvas, 0.6, 0, canvas)
    return canvas


def boxes(img):
    i = img.BGR
    canvas = np.zeros(i.shape, dtype='uint8')
    thick = round(2.4 * (i.shape[0] / 1280))
    for piece in img.pieces:
        x0, y0, x1, y1, conf, num, _ = piece
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        conf = round(float(conf), 2)
        num = str(int(num))
        color = COLORS[num]
        symbol = SYMBOLS[num]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color=color, thickness=thick)
        cv2.putText(canvas, f"{symbol} {conf}", (x0-5, y0-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, thick)

    cv2.addWeighted(i, 0.6, canvas, 0.8, 0, canvas)
    img.yolopieces = canvas
    return img
