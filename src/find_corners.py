import cv2
import numpy as np

import auxiliar as aux
import drawings as draw
import lffilter as lf

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
    while lv < 9 or lh < 9:
        minlen = max(minlen - 2, minlen0 / 2)
        tvotes -= 2
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
            vert, hori = magic_dir(img, vert, hori)
        lv, lh = len(vert), len(hori)
        ll = lv + lh
        print(f"{ll} # [{lv}][{lh}] @",
              f"{angle}º, {tvotes}, {minlen}, {maxgap}")

    canvas = draw.lines(img.gray3ch, vert, hori)
    aux.save(img, "hough_magic_final", canvas)
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
    aux.save(img, "corners", canvas)
    exit()

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
    lv, lh = len(vert), len(hori)
    distv, disth = get_distances(vert, hori)
    print("distances:")
    print("disth:\n", disth)
    print("distv:\n", distv)

    medv, medh = aux.mean_dist(distv, disth)
    print(f"{medv=}")
    print(f"{medh=}")

    print("removing for sure wrong vertical lines...")
    vert = aux.wrong_lines(vert, distv, medv, tol=2)
    lv = len(vert)
    print("removing for sure wrong horizontal lines...")
    hori = aux.wrong_lines(hori, disth, medh, tol=2)
    lh = len(hori)

    canvas = draw.lines(img.gray3ch, vert, hori)
    aux.save(img, "after_wrong", canvas)

    tol = 2
    ww = img.bwidth
    hh = img.bheigth
    changed = False
    if lv == 9:
        vtol = medv + tol + DX
        if abs(vert[0, 0] - 0) > vtol and abs(vert[0, 2] - 0) > vtol:
            vert = vert[0:-1]
            changed = True
        elif abs(vert[-1, 0] - ww) > vtol and abs(vert[-1, 2] - ww) > vtol:
            vert = vert[1:]
            changed = True
    if lh == 9:
        htol = medh + tol + DX
        if abs(hori[0, 1] - 0) > htol and abs(hori[0, 3] - 0) > htol:
            hori = hori[0:-1]
            changed = True
        elif abs(hori[-1, 1] - hh) > htol and abs(hori[-1, 3] - hh) > htol:
            hori = hori[1:]
            changed = True
    if changed:
        canvas = draw.lines(img.gray3ch, vert, hori)
        aux.save(img, "after==9", canvas)
        return vert, hori

    vert, hori = add_outer(vert, hori, medv, medh, img.bwidth, img.bheigth)
    return vert, hori


def get_distances(vert, hori):
    def _between(line2, line1):
        dist1 = min_distance(line1[0:2], line1[2:4], line2[0:2])
        dist2 = min_distance(line1[0:2], line1[2:4], line2[2:4])
        dist3 = min_distance(line2[0:2], line2[2:4], line1[0:2])
        dist4 = min_distance(line2[0:2], line2[2:4], line1[2:4])
        return min(dist1, dist2, dist3, dist4)

    def _get_dist(lines):
        dist = np.zeros((lines.shape[0], 2), dtype='int32')
        dist[0, 0] = dist[0, 1] = _between(lines[0], lines[1])
        i = 0
        for i in range(1, len(lines) - 1):
            dist[i, 0] = _between(lines[i-1], lines[i+0])
            dist[i, 1] = _between(lines[i+0], lines[i+1])
        i += 1
        dist[i, 0] = dist[i, 1] = _between(lines[i-1], lines[i])
        return dist

    return _get_dist(vert), _get_dist(hori)


def min_distance(A, B, E):
    # vector AB
    AB = [None, None]
    AB[0] = B[0] - A[0]
    AB[1] = B[1] - A[1]

    # vector BP
    BE = [None, None]
    BE[0] = E[0] - B[0]
    BE[1] = E[1] - B[1]

    # vector AP
    AE = [None, None]
    AE[0] = E[0] - A[0]
    AE[1] = E[1] - A[1]

    # variables to store dot product

    # calculating the dot product
    AB_BE = AB[0] * BE[0] + AB[1] * BE[1]
    AB_AE = AB[0] * AE[0] + AB[1] * AE[1]

    # minimum distance from
    # point E to the line segment
    reqAns = 0

    # case 1
    if (AB_BE > 0):
        # Finding the magnitude
        y = E[1] - B[1]
        x = E[0] - B[0]
        reqAns = np.sqrt(x*x + y*y)

    # case 2
    elif (AB_AE < 0):
        y = E[1] - A[1]
        x = E[0] - A[0]
        reqAns = np.sqrt(x*x + y*y)

    # Case 3
    else:
        # finding the perpendicular distance
        x1 = AB[0]
        y1 = AB[1]
        x2 = AE[0]
        y2 = AE[1]
        mod = np.sqrt(x1*x1 + y1*y1)
        reqAns = abs(x1*y2 - y1*x2) / mod

    return reqAns


def add_outer(vert, hori, medv, medh, ww, hh):
    tol = 2
    vtol = medv + tol + DX
    htol = medh + tol + DX
    print("adding missing outer lines...")
    while abs(vert[0, 0] - 0) > vtol and abs(vert[0, 2] - 0) > vtol:
        x1 = vert[0, 0] - abs(vert[0, 0] - vert[1, 0])
        y1 = vert[0, 1]
        x2 = vert[0, 2] - abs(vert[0, 2] - vert[1, 2])
        y2 = vert[0, 3]
        new = np.array([[x1, y1, x2, y2, 0, 0]], dtype='int32')
        vert = np.append(vert, new, axis=0)
        vert = vert[np.argsort(vert[:, 0])]
    while abs(vert[-1, 0] - ww) > vtol and abs(vert[-1, 2] - ww) > vtol:
        x1 = vert[-1, 0] + abs(vert[-1, 0] - vert[-2, 0])
        y1 = vert[-1, 1]
        x2 = vert[-1, 2] + abs(vert[-1, 2] - vert[-2, 2])
        y2 = vert[-1, 3]
        new = np.array([[x1, y1, x2, y2, 0, 0]], dtype='int32')
        vert = np.append(vert, new, axis=0)
        vert = vert[np.argsort(vert[:, 0])]
    while abs(hori[0, 1] - 0) > htol and abs(hori[0, 3] - 0) > htol:
        x1 = hori[0, 0]
        y1 = hori[0, 1] - abs(hori[0, 1] - hori[1, 1])
        x2 = hori[0, 2]
        y2 = hori[0, 3] - abs(hori[0, 3] - hori[1, 3])
        new = np.array([[x1, y1, x2, y2, 0, 0]], dtype='int32')
        hori = np.append(hori, new, axis=0)
        hori = hori[np.argsort(hori[:, 1])]
    while abs(hori[-1, 1] - hh) > htol and abs(hori[-1, 3] - hh) > htol:
        x1 = hori[-1, 0]
        y1 = hori[-1, 1] + abs(hori[-1, 1] - hori[-2, 1])
        x2 = hori[-1, 2]
        y2 = hori[-1, 3] + abs(hori[-1, 3] - hori[-2, 3])
        new = np.array([[x1, y1, x2, y2, 0, 0]], dtype='int32')
        hori = np.append(hori, new, axis=0)
        hori = hori[np.argsort(hori[:, 1])]

    return vert, hori
