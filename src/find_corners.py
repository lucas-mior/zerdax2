import cv2
import numpy as np

import auxiliar as aux
import drawings as draw
import lffilter as lf
import lines as li

from bundle_lines import bundle_lines

WLEN = 640
DX = 50


def find_corners(img):
    img = create_cannys(img)
    img = black_space(img)
    vert, hori = magic_lines(img)
    inter = aux.calc_intersections(img.gray3ch, vert, hori)
    canvas = draw.intersections(img.gray3ch, inter)
    # aux.save(img, "intersections", canvas)

    img.corners = calc_corners(img, inter)
    img = perspective_transform(img)

    return img


def create_cannys(img):
    print("finding edges for gray, S, V images...")
    cannyG = aux.find_edges(img, img.G, lowpass=lf.ffilter)
    cannyV = aux.find_edges(img, img.V, lowpass=lf.ffilter)
    # aux.save(img, "cannyG", cannyG)
    # aux.save(img, "cannyV", cannyV)
    img.canny = cv2.bitwise_or(cannyG, cannyV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img.canny = cv2.morphologyEx(img.canny, cv2.MORPH_DILATE, kernel)
    # aux.save(img, "canny_dilate", img.canny)
    img.canny = cv2.morphologyEx(img.canny, cv2.MORPH_CLOSE, kernel)
    # aux.save(img, "canny_closed", img.canny)
    return img


def magic_lines(img):
    print("finding all lines of board...")

    minlen0 = round((img.bwidth + img.bheigth) * 0.3)
    maxgap = round(minlen0 / 4)
    tvotes = round(minlen0 / 2)
    angle = 1  # degrees
    tangle = np.deg2rad(angle)  # radians

    gotmin = False
    minlen = 0
    for minlen in range(minlen0, round(minlen0 * 0.6), -16):
        tvotes = round(minlen / 2)
        print(f"trying @{angle}º, {tvotes}, {minlen}, {maxgap}")
        lines = cv2.HoughLinesP(img.canny, 1,
                                tangle, tvotes, None, minlen, maxgap)
        if lines is not None and len(lines) >= 10:
            lines = lines[:, 0, :]
            gotmin = True
            break
    if not gotmin:
        print("magic_lines() failed @ {angle}º, {tvotes}, {minlen}, {maxgap}")
        # aux.save(img, "lastcanny", img.canny)
        canvas = draw.lines(img.gray3ch, lines)
        exit(1)

    print(f"{minlen=}")
    minlen0 = minlen
    ll = lv = lh = 0
    while lv < 9 or lh < 9 and tvotes > minlen0 / 3:
        minlen = max(minlen - 2, minlen0 / 1.5)
        tvotes -= 5
        lines = cv2.HoughLinesP(img.canny, 1,
                                tangle, tvotes, None, minlen, maxgap)
        if ll := len(lines) < 12:
            print(f"{ll} @ {angle}, {tvotes}, {minlen}, {maxgap}")
            continue
        lines = lines[:, 0, :]
        lines = bundle_lines(lines)
        lines = aux.radius_theta(lines)
        vert, hori = split_lines(img, lines)
        vert, hori = filter_angles(vert, hori)
        if (lv := len(vert)) >= 6 <= (lh := len(hori)):
            vert = vert[np.argsort(vert[:, 0])]
            hori = hori[np.argsort(hori[:, 1])]
            vert, hori, medv, medh = magic_dir(img, vert, hori)
            vert, hori = li.rem_1011(img, vert, hori, medv, medh)
        lv, lh = len(vert), len(hori)
        ll = lv + lh
        print(f"{ll} # [{lv}][{lh}] @",
              f"{angle}º, {tvotes}, {minlen}, {maxgap}")

    if lv < 9 or lh < 9:
        vert, hori = li.add_outer(vert, hori, medv, medh,
                                  img.bwidth, img.bheigth)
        vert, hori = li.add_middle(vert, hori, medv, medh)
    if lv > 9 or lh > 9:
        vert, hori = li.remove_extras(vert, hori)
        vert, hori = li.add_last_outer(vert, hori, medv, medh)

    canvas = draw.lines(img.gray3ch, vert, hori)
    # aux.save(img, "hough_magic_final", canvas)
    return vert, hori


def filter_angles(vert, hori, tol=15):
    def _filter(lines):
        rem = np.zeros(lines.shape[0], dtype='uint8')
        angle = np.median(lines[:, 5])

        for i, line in enumerate(lines):
            x1, y1, x2, y2, r, t = line
            if abs(t - angle) > 15:
                rem[i] = 1
            else:
                rem[i] = 0
        return lines[rem == 0]

    return _filter(vert), _filter(hori)


def calc_corners(img, inter):
    print("calculating 4 corners of board...")
    psum = np.zeros((inter.shape[0], 3), dtype='int32')
    psub = np.zeros((inter.shape[0], 3), dtype='int32')

    psum[:, 0] = inter[:, 0]
    psum[:, 1] = inter[:, 1]
    psum[:, 2] = inter[:, 0] + inter[:, 1]
    psub[:, 0] = inter[:, 0]
    psub[:, 1] = inter[:, 1]
    psub[:, 2] = inter[:, 0] - inter[:, 1]

    BR = psum[np.argmax(psum[:, 2])][0:2]
    TR = psub[np.argmax(psub[:, 2])][0:2]
    BL = psub[np.argmin(psub[:, 2])][0:2]
    TL = psum[np.argmin(psum[:, 2])][0:2]

    BR, BL, TR, TL = broad_corners(img, BR, BL, TR, TL)

    canvas = draw.corners(img.gray3ch, BR, BL, TR, TL)
    # aux.save(img, "corners", canvas)

    return np.array([BR, BL, TR, TL], dtype='int32')


def perspective_transform(img):
    print("transforming perspective...")
    BR = img.corners[0]
    BL = img.corners[1]
    TR = img.corners[2]
    TL = img.corners[3]
    orig_points = np.array(((TL[0], TL[1]), (TR[0], TR[1]),
                            (BR[0], BR[1]), (BL[0], BL[1])), dtype="float32")

    width = WLEN
    height = WLEN
    img.wwidth = width
    img.wheigth = height

    newshape = np.array([[0, 0], [width-1, 0],
                        [width-1, height-1], [0, height-1]], dtype="float32")
    print("creating transform matrix...")
    img.warpMatrix = cv2.getPerspectiveTransform(orig_points, newshape)
    _, img.warpInvMatrix = cv2.invert(img.warpMatrix)
    print("warping image...")
    img.wg = cv2.warpPerspective(img.G, img.warpMatrix, (width, height))
    img.wv = cv2.warpPerspective(img.V, img.warpMatrix, (width, height))
    # aux.save(img, "warpclaheG", img.wg)
    # aux.save(img, "warpclaheV", img.wv)

    return img


def split_lines(img, lines):
    lines = aux.radius_theta(lines)
    lines = np.array(lines, dtype='float32')
    if (lines.shape[1] < 6):
        lines = aux.radius_theta(lines)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compact, labels, centers = cv2.kmeans(lines[:, 5], 3, None,
                                          criteria, 10, flags)
    labels = np.ravel(labels)

    A = lines[labels == 0]
    B = lines[labels == 1]

    d1 = abs(centers[0] - centers[1])
    d2 = abs(centers[0] - centers[2])
    d3 = abs(centers[1] - centers[2])

    dd1 = d1 < 22.5 and d2 > 22.5 and d3 > 22.5
    dd2 = d2 < 22.5 and d1 > 22.5 and d3 > 22.5
    dd3 = d3 < 22.5 and d1 > 22.5 and d2 > 22.5

    if dd1 or dd2 or dd3:
        compact, labels, centers = cv2.kmeans(lines[:, 5], 2, None,
                                              criteria, 10, flags)

        labels = np.ravel(labels)
        A = lines[labels == 0]
        B = lines[labels == 1]

    if len(centers) == 3:
        # redo kmeans using absolute inclination
        lines = aux.radius_theta(lines, abs_angle=True)
        lines = np.array(lines, dtype='float32')
        compact, labels, centers = cv2.kmeans(lines[:, 5], 2, None,
                                              criteria, 10, flags)
        labels = np.ravel(labels)
        A = lines[labels == 0]
        B = lines[labels == 1]

    if centers[1] < centers[0]:
        return np.int32(A), np.int32(B)
    else:
        return np.int32(B), np.int32(A)


def broad_corners(img, BR, BL, TR, TL):
    print("adding margin for corners...")
    BR[0] = min(img.bwidth-1,  BR[0]+8)
    BR[1] = min(img.bheigth-1, BR[1]+8)
    BL[0] = max(0,             BL[0]-8)
    BL[1] = min(img.bheigth-1, BL[1]+8)
    TR[0] = min(img.bwidth-1,  TR[0]+8)
    TR[1] = max(0,             TR[1]-8)
    TL[0] = max(0,             TL[0]-8)
    TL[1] = max(0,             TL[1]-8)
    return BR, BL, TR, TL


def make_border(image):
    return cv2.copyMakeBorder(image, DX, DX, DX, DX,
                              cv2.BORDER_CONSTANT, None, value=0)


def black_space(img):
    img.board = make_border(img.board)
    img.gray = make_border(img.gray)
    img.G = make_border(img.G)
    img.V = make_border(img.V)
    img.gray3ch = make_border(img.gray3ch)
    img.canny = make_border(img.canny)

    img.bwidth += (DX*2)
    img.bheigth += (DX*2)
    return img


def magic_dir(img, vert, hori):
    def _check_save(title):
        nonlocal lv, lh, vert, hori
        if lv != len(vert) or lh != len(hori):
            canvas = draw.lines(img.warp3ch, vert, hori)
            # aux.save(img, title, canvas)
            lv, lh = len(vert), len(hori)
        return

    lv, lh = len(vert), len(hori)
    distv, disth = li.get_distances(vert, hori)
    medv, medh = li.mean_dist(distv, disth)

    print("removing for sure wrong vertical lines...")
    vert = li.wrong_lines(vert, distv, medv, tol=2)
    lv = len(vert)
    print("removing for sure wrong horizontal lines...")
    hori = li.wrong_lines(hori, disth, medh, tol=2)
    lh = len(hori)
    _check_save("rem_wrong")
    return vert, hori, medv, medh
