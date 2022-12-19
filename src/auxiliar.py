import cv2
import numpy as np
import math

# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt

i = 1


def logprint(img, text):
    if img.log:
        print(text)


def determinant(a, b):
    return a[0]*b[1] - a[1]*b[0]


def save(img, filename, image):
    global i
    # cv2.imwrite(f"{img.basename}{i:02d}_{filename}.png", image)
    i += 1


def savefig(img, filename, fig):
    global i
    fig.savefig(f"{img.basename}{i:02d}_{filename}.png")
    i += 1


def radius(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    radius = math.sqrt(dx*dx + dy*dy)
    return round(radius)


def theta(x1, y1, x2, y2, absol=False):
    if absol:
        orientation = math.atan2(abs(y1-y2), abs(x2-x1))
    else:
        orientation = math.atan2(y1-y2, x2-x1)
    orientation = math.degrees(orientation)
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
        canvas1 = np.zeros(img.warped3ch.shape, dtype='uint8')
        canvas2 = np.zeros(img.warped3ch.shape, dtype='uint8')
    else:
        canvas1 = np.zeros(img.gray3ch.shape, dtype='uint8')
        canvas2 = np.zeros(img.gray3ch.shape, dtype='uint8')

    for x1, y1, x2, y2, r, t in vert:
        cv2.line(canvas1, (x1, y1), (x2, y2),
                 color=(255, 0, 0), thickness=3)
    for x1, y1, x2, y2, r, t in hori:
        cv2.line(canvas1, (x1, y1), (x2, y2),
                 color=(0, 255, 0), thickness=3)

    if warp:
        canvas2 = cv2.addWeighted(img.warped3ch, 0.5, canvas1, 0.5, 1)
    else:
        canvas2 = cv2.addWeighted(img.gray3ch, 0.5, canvas1, 0.5, 1)
    save(img, name, canvas2)


def find_canny(img, image, wmin=5, c_thrh=220):
    logprint(img, f"finding edges with Canny until mean" ">= {wmin:0=.1f}...")

    def lp(sign):
        logprint(img, f"{w:0=.2f} {sign} {wmin:0=.1f}, @ {c_thrl}, {c_thrh}")
        return

    got_canny = False
    ctmin = 30
    while c_thrh >= ctmin:
        c_thrl = max(10, round(c_thrh*0.8))
        clmin = 10
        while c_thrl >= clmin:
            canny = cv2.Canny(image, c_thrl, c_thrh)
            w = canny.mean()
            if w > wmin:
                lp("<")
                got_canny = True
                break

            lp("<")

            gain = wmin - w
            diff = round(max(8, gain*10))
            if c_thrl <= clmin:
                break
            c_thrl = max(clmin, c_thrl - diff)

        if got_canny:
            break

        if c_thrh <= ctmin:
            break
        diff = round(max(5, gain*(c_thrh/15)))
        c_thrh = max(ctmin, c_thrh - diff)

    if not got_canny:
        if diff > 2:
            print(f"Canny failed @ {c_thrl}, {c_thrh}")
            exit(1)
        else:
            print(f"Canny failed, but trying anyway")

    return canny, c_thrh


def auto_canny(image, sigma=0.6):
    """
    Canny edge detection with automatic thresholds.
    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged
