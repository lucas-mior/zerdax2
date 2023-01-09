import cv2
import numpy as np
import drawings as draw
import lffilter as lf

i = 1


def save(img, filename, image):
    global i
    cv2.imwrite(f"{img.basename}{i:02d}_{filename}.png", image)
    i += 1


def radius(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    radius = np.sqrt(dx*dx + dy*dy)
    return round(radius)


def theta(x1, y1, x2, y2, abs_angle=False):
    if abs_angle:
        angle = np.arctan2(abs(y1-y2), abs(x2-x1))
    else:
        angle = np.arctan2(y1-y2, x2-x1)

    angle = np.rad2deg(angle)
    return round(angle)


def radius_theta(lines, abs_angle=False):
    dummy = np.zeros((lines.shape[0], 6), dtype='int32')
    dummy[:, 0:4] = lines[:, 0:4]
    lines = dummy[np.argsort(dummy[:, 0])]

    for i, line in enumerate(lines):
        x1, y1, x2, y2, r, t = line
        lines[i, 4] = radius(x1, y1, x2, y2)
        lines[i, 5] = theta(x1, y1, x2, y2, abs_angle=abs_angle)

    return lines


def geo_lines(lines):
    if (lines.shape[1] < 6):
        lines = radius_theta(lines, abs_angle=True)
    vert = lines[abs(lines[:, 5]) > 45]
    hori = lines[abs(lines[:, 5]) < 45]

    vert = vert[np.argsort(vert[:, 0])]
    hori = hori[np.argsort(hori[:, 1])]

    return vert, hori


def gauss(image):
    filtered = cv2.GaussianBlur(image, (5, 5), 0.5)
    return filtered


def find_edges(img, image, lowpass):
    print("filtering image...")
    image = lowpass(image)
    # save(img, "lowpass", image)
    if lowpass == lf.ffilter:
        wmin = 10
        thigh = 200
    elif lowpass == gauss:
        wmin = 12
        thigh = 230
    else:
        print(f"which lowfilter is {lowpass}?")
        exit(1)
    return find_canny(image, wmin, thigh)


def find_canny(image, wmin=8, thigh=230):
    print(f"finding edges with Canny until mean >= {wmin:0=.1f}...")

    got_canny = False
    thighmin = 30
    tlowmin = 10
    while thigh >= thighmin:
        tlow = max(tlowmin, round(thigh*0.8))
        while tlow >= tlowmin:
            canny = cv2.Canny(image, tlow, thigh)
            w = round(np.mean(canny), 2)
            if w >= wmin:
                print(f"{w} >= {wmin:0=.1f}, @ {tlow}, {thigh}")
                got_canny = True
                break
            else:
                print(f"{w} < {wmin:0=.1f}, @ {tlow}, {thigh}")
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
        print("Canny failed, but trying anyway")

    return canny


def calc_intersections(image, lines1, lines2=None):
    print("calculating intersections...")

    def _det(a, b):
        return a[0]*b[1] - a[1]*b[0]

    inter = []
    corn = False
    if lines2 is None:
        corn = True
        lines2 = lines1

    for x1, y1, x2, y2, r, t in lines1:
        for xx1, yy1, xx2, yy2, rr, tt in lines2:
            close = False
            if (x1, y1) == (xx1, yy1) and (x2, y2) == (xx2, yy2):
                continue

            dtheta = abs(t - tt)
            if corn and (dtheta < 20 or dtheta > 160):
                # print(f"t - tt: {dtheta}")
                continue

            xdiff = (x1 - x2, xx1 - xx2)
            ydiff = (y1 - y2, yy1 - yy2)

            div = _det(xdiff, ydiff)
            if div == 0:
                print("div == 0")
                continue

            d = (_det((x1, y1), (x2, y2)),
                 _det((xx1, yy1), (xx2, yy2)))
            x = round(_det(d, xdiff) / div)
            y = round(_det(d, ydiff) / div)

            if x >= image.shape[1] or y >= image.shape[0]:
                print(f"{x} >= {image.shape[1]} or {y} >= {image.shape[0]}")
                continue
            if x < 0 or y < 0:
                print(f"{x} < 0 or {y} < 0")
                continue

            for p in inter:
                if radius(x, y, p[0], p[1]) < 10:
                    close = True
                    break
            if close:
                continue
            else:
                inter.append((x, y))

    return np.array(inter, dtype='int32')
