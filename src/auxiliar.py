import cv2
import numpy as np
from numpy.linalg import det
import lffilter as lf

i = 1


def save(img, filename, image):
    global i
    title = f"{img.basename}{i:02d}_{filename}.png"
    print(f"saving {title}...")
    cv2.imwrite(title, image)
    i += 1


def radius(line):
    x1, y1, x2, y2 = line[:4]
    dx = x2 - x1
    dy = y2 - y1
    return np.sqrt(dx*dx + dy*dy)


def theta(line, abs_angle=False):
    x1, y1, x2, y2 = line[:4]
    if abs_angle:
        angle = np.arctan2(abs(y1-y2), abs(x2-x1))
    else:
        if x2 < x1:
            a1, b1 = x1, y1
            x1, y1 = x2, y2
            x2, y2 = a1, b1
        angle = np.arctan2(y1-y2, x2-x1)

    return np.rad2deg(angle)


def radius_theta(vert, hori=None, abs_angle=False):
    def _create(lines):
        dummy = np.zeros((lines.shape[0], 7), dtype='int32')
        dummy[:, 0:4] = lines[:, 0:4]
        lines = dummy[np.argsort(dummy[:, 0])]

        for i, line in enumerate(lines):
            x1, y1, x2, y2, r, t, _ = line
            lines[i, 4] = radius((x1, y1, x2, y2))
            lines[i, 5] = theta((x1, y1, x2, y2), abs_angle=abs_angle)
        return np.round(lines)

    if hori is not None:
        hori = _create(hori)
    vert = _create(vert)
    return vert, hori


def gauss(image):
    filtered = cv2.GaussianBlur(image, (5, 5), 0.5)
    return filtered


def find_edges(img, image, lowpass, bonus=0):
    print("filtering image...")
    image = lowpass(image)
    pbonus = len(img.pieces) / 15
    if lowpass == lf.ffilter:
        wmin = 10.5 + pbonus + bonus
        thigh0 = 250
    elif lowpass == gauss:
        wmin = 12 + pbonus + bonus
        thigh0 = 220
    canny, got_canny = find_canny(image, wmin, thigh0)
    if not got_canny:
        save(img, "lowpass", image)
    return canny, got_canny


def find_canny(image, wmin=8, thigh0=250):
    print(f"finding edges with Canny until mean >= {wmin:0=.1f}...")

    got_canny = False
    thighmin = 30
    tlowmin = 10
    thigh = thigh0
    while thigh >= thighmin:
        tlow = max(tlowmin, round(thigh*0.8))
        while tlow >= tlowmin:
            canny = cv2.Canny(image, tlow, thigh)
            w = np.mean(canny)
            if w >= wmin:
                print(f"{w:0=.2f} >= {wmin:0=.1f}, @ {tlow}, {thigh}")
                got_canny = True
                break
            else:
                print(f"{w:0=.2f} < {wmin:0=.1f}, @ {tlow}, {thigh}")
                gain = wmin - w
                diff = round(max(8, gain*8))
                if tlow <= tlowmin:
                    break
                tlow = max(tlowmin, tlow - diff)

        if got_canny or (thigh <= thighmin):
            break
        else:
            diff = round(max(5, gain*(thigh/20)))
            thigh = max(thighmin, thigh - diff)

    if not got_canny:
        print(f"Failed to find edges with mean >= {wmin:0=.1f}")

    return canny, got_canny


def calc_intersections(image, lines1, lines2=None):
    print("calculating intersections between group(s) of lines...")

    if lines2 is None:
        lines2 = lines1

    rows = []
    for x1, y1, x2, y2, r, t, _ in lines1:
        col = []
        for xx1, yy1, xx2, yy2, rr, tt, _ in lines2:
            if (x1, y1) == (xx1, yy1) and (x2, y2) == (xx2, yy2):
                continue

            dtheta = abs(t - tt)
            if (dtheta < 20 or dtheta > 160):
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
    print("calculating intersections between 2 lines...")
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
        print("lines should not be equal")
        return (30000, 30000)

    xdiff = (x1 - x2, xx1 - xx2)
    ydiff = (y1 - y2, yy1 - yy2)

    div = det([xdiff, ydiff])
    if div == 0:
        print("div == 0 (parallel lines)")
        return (30000, 30000)

    d = (det([(x1, y1), (x2, y2)]),
         det([(xx1, yy1), (xx2, yy2)]))
    x = round(det([d, xdiff]) / div)
    y = round(det([d, ydiff]) / div)
    return np.array((x, y), dtype='int32')
