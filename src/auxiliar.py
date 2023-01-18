import cv2
import numpy as np
from numpy.linalg import det
import logging as log
import lffilter as lf
import constants as consts

i = 1


def save(filename, image):
    global i
    title = f"z{i:04d}_{filename}.png"
    print(f"saving {title}...")
    cv2.imwrite(title, image)
    i += 1


def calc_intersections(lines1, lines2=None):
    log.info("calculating intersections between group(s) of lines...")

    if lines2 is None:
        lines2 = lines1

    rows = []
    for x1, y1, x2, y2, r, t, _ in lines1:
        col = []
        for xx1, yy1, xx2, yy2, rr, tt, _ in lines2:
            if (x1, y1) == (xx1, yy1) and (x2, y2) == (xx2, yy2):
                continue

            dtheta = abs(t - tt)
            tol1 = consts.min_angle_to_intersect
            tol2 = 180 - tol1
            if (dtheta < tol1 or dtheta > tol2):
                continue

            xdiff = (x1 - x2, xx1 - xx2)
            ydiff = (y1 - y2, yy1 - yy2)

            div = det([xdiff, ydiff])
            if div == 0:
                continue

            d = (det([(x1, y1), (x2, y2)]),
                 det([(xx1, yy1), (xx2, yy2)]))
            x = det([d, xdiff]) / div
            y = det([d, ydiff]) / div
            col.append((x, y))
        rows.append(col)

    inter = np.round(rows)
    return np.array(inter, dtype='int32')


def calc_intersection(line, ww=500, hh=300, kind=0):
    log.debug("calculating intersections between 2 lines...")
    if kind == 0:
        line2 = (50, 0, 400, 0, 0, 0)
    elif kind == 1:
        line2 = (0, 50, 0, 400, 0, 0)
    elif kind == 2:
        line2 = (50, hh, 400, hh, 0, 0)
    elif kind == 3:
        line2 = (ww, 50, ww, 400, 0, 0)

    x1, y1, x2, y2 = line[:4]
    xx1, yy1, xx2, yy2 = line2[:4]
    if (x1, y1, x2, x2) == (xx1, yy1, xx2, yy2):
        log.warning("lines should not be equal")
        return (30000, 30000)

    xdiff = (x1 - x2, xx1 - xx2)
    ydiff = (y1 - y2, yy1 - yy2)

    div = det([xdiff, ydiff])
    if div == 0:
        log.warning("div == 0 (parallel lines)")
        return (30000, 30000)

    d = (det([(x1, y1), (x2, y2)]),
         det([(xx1, yy1), (xx2, yy2)]))
    x = round(det([d, xdiff]) / div)
    y = round(det([d, ydiff]) / div)
    return np.array((x, y), dtype='int32')


def debugging():
    return log.root.level < 20
