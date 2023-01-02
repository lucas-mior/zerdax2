import cv2
import numpy as np

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
    dummy = np.zeros((lines.shape[0], 6), dtype='int32')
    dummy[:, 0:4] = np.copy(lines[:, 0:4])
    lines = dummy
    lines = lines[np.argsort(lines[:, 0])]

    for i, line in enumerate(lines):
        x1, y1, x2, y2, r, t = line
        lines[i, 4] = radius(x1, y1, x2, y2)
        lines[i, 5] = theta(x1, y1, x2, y2, absol=absol)

    return lines


def geo_lines(lines):
    lines = radius_theta(lines)
    vert = lines[abs(lines[:, 5]) > 45]
    hori = lines[abs(lines[:, 5]) < 45]

    vert = vert[np.argsort(vert[:, 0])]
    hori = hori[np.argsort(hori[:, 1])]

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


def find_canny(image, wmin=5, thigh=220):
    print(f"finding edges with Canny until mean >= {wmin:0=.1f}...")

    got_canny = False
    thighmin = 30
    while thigh >= thighmin:
        tlow = max(10, round(thigh*0.8))
        tlowmin = 10
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
                diff = round(max(8, gain*10))
                if tlow <= tlowmin:
                    break
                tlow = max(tlowmin, tlow - diff)

        if got_canny or (thigh <= thighmin):
            break
        else:
            diff = round(max(5, gain*(thigh/15)))
            thigh = max(thighmin, thigh - diff)

    if not got_canny:
        print("Canny failed, but trying anyway")

    return canny


def draw_intersections(image, inter):
    canvas = np.zeros(image.shape, dtype='uint8')
    for i, p in enumerate(inter):
        cv2.circle(canvas, p, radius=5,
                   color=(30+i*2, 0, 225-i*2), thickness=-1)
    cv2.addWeighted(image, 0.6, canvas, 0.4, 0, canvas)
    return canvas
