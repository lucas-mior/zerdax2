import numpy as np
from numpy import linalg
import logging as log

import constants


def calculate_extern(lines0, lines1=None):
    log.debug("calculating external intersections between groups of lines...")

    if lines1 is None:
        log.debug("-> calculating all intersections in the same group")
        lines1 = lines0
    if lines0 is None:
        log.debug("-> line group is empty, returning None.")
        return None

    rows = []
    for i in range(l0 := lines0.shape[0]):
        x0, y0, x1, y1, r, t = lines0[i]
        col = []
        for j in range(l1 := lines1.shape[0]):
            xx0, yy0, xx1, yy1, rr, tt = lines1[j]
            if 0 != i != (l0-1) and 0 != j != (l1-1):
                col.append((30000, 30000))
                continue
            if (x0, y0, x1, x1) == (xx0, yy0, xx1, yy1):
                continue

            dtheta = abs(t - tt)
            tol0 = constants.min_angle_to_intersect
            tol1 = 180*100 - tol0
            if (dtheta < tol0 or dtheta > tol1):
                continue

            xdiff = (x0 - x1, xx0 - xx1)
            ydiff = (y0 - y1, yy0 - yy1)

            div = linalg.det([xdiff, ydiff])
            if div == 0:
                continue

            d = (linalg.det([(x0, y0), (x1, y1)]),
                 linalg.det([(xx0, yy0), (xx1, yy1)]))
            x = linalg.det([d, xdiff]) / div
            y = linalg.det([d, ydiff]) / div
            col.append((x, y))
        rows.append(col)

    try:
        inter = np.round(rows)
        return np.array(inter, dtype='int32')
    except Exception:
        return None


def calculate_all(lines0, lines1=None, onlylast=False, limit=False):
    if lines1 is None:
        lines1 = lines0

    max_x0 = np.max([lines0[:, 0], lines0[:, 2]])
    max_x1 = np.max([lines1[:, 0], lines1[:, 2]])
    max_y0 = np.max([lines0[:, 1], lines0[:, 3]])
    max_y1 = np.max([lines1[:, 1], lines1[:, 3]])
    max_x = max(max_x0, max_x1)
    max_y = max(max_y0, max_y1)

    rows = []
    for line0 in lines0:
        x0, y0, x1, y1, r, t = line0
        col = []
        for line1 in lines1:
            xx0, yy0, xx1, yy1, rr, tt = line1
            if (x0, y0, x1, x1) == (xx0, yy0, xx1, yy1):
                continue

            if not limit:
                dtheta = abs(t - tt)
                tol0 = constants.min_angle_to_intersect
                tol1 = 180*100 - tol0
                if (dtheta < tol0 or dtheta > tol1):
                    continue

            xdiff = (x0 - x1, xx0 - xx1)
            ydiff = (y0 - y1, yy0 - yy1)

            div = linalg.det([xdiff, ydiff])
            if div == 0:
                continue

            d = (linalg.det([(x0, y0), (x1, y1)]),
                 linalg.det([(xx0, yy0), (xx1, yy1)]))
            x = linalg.det([d, xdiff]) / div
            y = linalg.det([d, ydiff]) / div
            if limit and (x < 0 or x > max_x or y < 0 or y > max_y):
                continue
            col.append((x, y))
        rows.append(col)

    inter = np.round(rows)
    return np.array(inter, dtype='int32')


def calculate_single(line0, canny, kind=0):
    if kind == 0:
        line1 = (50, 0, 400, 0, 0, 0)
    elif kind == 1:
        line1 = (0, 50, 0, 400, 0, 0)
    elif kind == 2:
        line1 = (50, canny.shape[0], 400, canny.shape[0], 0, 0)
    elif kind == 3:
        line1 = (canny.shape[1], 50, canny.shape[1], 400, 0, 0)

    x0, y0, x1, y1 = line0[:4]
    xx0, yy0, xx1, yy1 = line1[:4]
    if (x0, y0, x1, x1) == (xx0, yy0, xx1, yy1):
        log.warning("lines should not be equal")
        return (30000, 30000)

    xdiff = (x0 - x1, xx0 - xx1)
    ydiff = (y0 - y1, yy0 - yy1)

    div = linalg.det([xdiff, ydiff])
    if div == 0:
        log.warning("div == 0 (parallel lines)")
        return (30000, 30000)

    d = (linalg.det([(x0, y0), (x1, y1)]),
         linalg.det([(xx0, yy0), (xx1, yy1)]))

    x = round(linalg.det([d, xdiff]) / div)
    y = round(linalg.det([d, ydiff]) / div)
    return np.array((x, y), dtype='int32')


def shorten(inters, canny):
    i0 = calculate_single(inters, canny, 0)
    i1 = calculate_single(inters, canny, 1)
    i2 = calculate_single(inters, canny, 2)
    i3 = calculate_single(inters, canny, 3)
    inters = np.array([i0, i1, i2, i3], dtype='int32')

    image_width = canny.shape[1]
    image_height = canny.shape[0]
    inters = inters[(inters[:, 0] >= 0) & (inters[:, 1] >= 0) &
                    (inters[:, 0] <= image_width) &
                    (inters[:, 1] <= image_height)]
    return inters
