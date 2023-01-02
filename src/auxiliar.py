import cv2
import numpy as np

# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt

i = 1


def determinant(a, b):
    return a[0]*b[1] - a[1]*b[0]


def save(img, filename, image):
    global i
    cv2.imwrite(f"{img.basename}{i:02d}_{filename}.png", image)
    i += 1


def radius(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    radius = np.sqrt(dx*dx + dy*dy)
    return round(radius)


def theta(x1, y1, x2, y2, absol=False):
    if absol:
        orientation = np.arctan2(abs(y1-y2), abs(x2-x1))
    else:
        orientation = np.arctan2(y1-y2, x2-x1)
    orientation = np.rad2deg(orientation)
    if abs(orientation) > 90:
        print(f"theta({x1}, {y1}, {x2}, {y2})")
        print("orientation:", orientation)
        exit(1)

    return round(orientation)


def radius_theta(lines, absol=False):
    dummy = np.zeros((lines.shape[0], 1, 6), dtype='int32')
    dummy[:, 0, 0:4] = np.copy(lines[:, 0, 0:4])
    lines = dummy
    lines = lines[lines[:, 0, 0].argsort()]
    i = 0
    for line in lines:
        for x1, y1, x2, y2, r, t in line:
            lines[i, 0, 4] = radius(x1, y1, x2, y2)
            lines[i, 0, 5] = theta(x1, y1, x2, y2, absol=absol)
            i += 1

    return lines


def geo_lines(lines):
    lines = radius_theta(lines)
    vert = lines[abs(lines[:, 0, 5]) > 45]
    hori = lines[abs(lines[:, 0, 5]) < 45]

    vert = vert[vert[:, 0, 0].argsort()]
    hori = hori[hori[:, 0, 1].argsort()]

    return vert, hori


def save_lines(img, name, vert, hori, warp=True):
    if warp:
        canvas = np.zeros(img.warp3ch.shape, dtype='uint8')
    else:
        canvas = np.zeros(img.gray3ch.shape, dtype='uint8')

    for x1, y1, x2, y2, r, t in vert:
        cv2.line(canvas, (x1, y1), (x2, y2),
                 color=(255, 0, 0), thickness=img.thick)
    for x1, y1, x2, y2, r, t in hori:
        cv2.line(canvas, (x1, y1), (x2, y2),
                 color=(0, 255, 0), thickness=img.thick)

    if warp:
        cv2.addWeighted(img.warp3ch, 0.6, canvas, 0.4, 0, canvas)
    else:
        cv2.addWeighted(img.gray3ch, 0.6, canvas, 0.4, 0, canvas)
    save(img, name, canvas)
    return


def find_canny(img, image, wmin=5, thigh=220):
    print(f"finding edges with Canny until mean >= {wmin:0=.1f}...")

    def lp(sign):
        print(f"{w:0=.2f} {sign} {wmin:0=.1f}, @ {tlow}, {thigh}")
        return

    got_canny = False
    ctmin = 30
    while thigh >= ctmin:
        tlow = max(10, round(thigh*0.8))
        clmin = 10
        while tlow >= clmin:
            canny = cv2.Canny(image, tlow, thigh)
            w = np.mean(canny)
            if w > wmin:
                lp("<")
                got_canny = True
                break

            lp("<")

            gain = wmin - w
            diff = round(max(8, gain*10))
            if tlow <= clmin:
                break
            tlow = max(clmin, tlow - diff)

        if got_canny:
            break

        if thigh <= ctmin:
            break
        diff = round(max(5, gain*(thigh/15)))
        thigh = max(ctmin, thigh - diff)

    if not got_canny:
        print("Canny failed, but trying anyway")

    return canny, thigh
