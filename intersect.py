import numpy as np
from numpy import linalg
import logging as log

MIN_ANGLE_TO_INTERSECT = 20 * 100


def calculate_extern(hori, vert=None):
    log.debug("calculating external intersections between groups of lines...")

    if vert is None:
        log.debug("-> calculating all intersections in the same group")
        vert = hori
    if hori is None:
        log.debug("-> line group is empty, returning None.")
        return None

    rows = []
    for i in range(l0 := hori.shape[0]):
        x0, y0, x1, y1, r, t = hori[i]
        column = []
        for j in range(l1 := vert.shape[0]):
            xx0, yy0, xx1, yy1, rr, tt = vert[j]
            if 0 != i != (l0 - 1) and 0 != j != (l1 - 1):
                column.append((30000, 30000))
                continue
            if (x0, y0, x1, x1) == (xx0, yy0, xx1, yy1):
                continue

            dtheta = abs(t - tt)
            tol0 = MIN_ANGLE_TO_INTERSECT
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
            column.append((x, y))
        rows.append(column)

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
                tol0 = MIN_ANGLE_TO_INTERSECT
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
            if limit:
                if x < 0 or x > max_x or y < 0 or y > max_y:
                    continue
            col.append((x, y))
        rows.append(col)

    inter = np.round(rows)
    return np.array(inter, dtype='int32')


def calculate_single(line0, canny, kind=0):
    match kind:
        case 0:
            line1 = (50, 0, 400, 0, 0, 0)
        case 1:
            line1 = (0, 50, 0, 400, 0, 0)
        case 2:
            line1 = (50, canny.shape[0], 400, canny.shape[0], 0, 0)
        case 3:
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
        log.debug("div == 0 (parallel lines)")
        return (30000, 30000)

    d = (linalg.det([(x0, y0), (x1, y1)]),
         linalg.det([(xx0, yy0), (xx1, yy1)]))

    x = round(linalg.det([d, xdiff]) / div)
    y = round(linalg.det([d, ydiff]) / div)
    return np.array((x, y), dtype='int32')
