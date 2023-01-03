import cv2
import numpy as np

import auxiliar as aux
import drawings as draw

from bundle_lines import bundle_lines

WARP_LEN = 640
DX = 40


def find_corners(img):
    img = black_space(img)
    lines = magic_lines(img)
    inter = calc_intersections(img, lines)
    img.corners = calc_corners(img, inter)
    img = perspective_transform(img)

    return img


def create_cannys(img, w=5, thighg=200, thighv=200, saveny=False):
    print("finding edges for gray, S, V images...")
    cannyG = aux.find_canny(img.G, wmin=w, thigh=thighg)
    cannyV = aux.find_canny(img.V, wmin=w, thigh=thighv)
    aux.save(img, "cannyG", cannyG)
    aux.save(img, "cannyV", cannyV)
    img.canny = cv2.bitwise_or(cannyG, cannyV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img.canny = cv2.morphologyEx(img.canny, cv2.MORPH_CLOSE, kernel)
    return img


def calc_intersections(img, lines):
    print("calculating intersections...")
    inter = []

    for x1, y1, x2, y2, r, t in lines:
        for xx1, yy1, xx2, yy2, rr, tt in lines:
            if (x1, y1) == (xx1, yy1) and (x2, y2) == (xx2, yy2):
                continue

            dtheta = abs(t - tt)
            if dtheta < 20 or dtheta > 160:
                # print(f"t - tt: {dtheta)")
                continue

            xdiff = (x1 - x2, xx1 - xx2)
            ydiff = (y1 - y2, yy1 - yy2)

            div = aux.determinant(xdiff, ydiff)
            if div == 0:
                print("div == 0")
                continue

            d = (aux.determinant((x1, y1), (x2, y2)),
                 aux.determinant((xx1, yy1), (xx2, yy2)))
            x = round(aux.determinant(d, xdiff) / div)
            y = round(aux.determinant(d, ydiff) / div)

            if x > img.bwidth or y > img.bheigth or x < 0 or y < 0:
                continue
            else:
                inter.append((x, y))

    inter = np.array(inter, dtype='int32')
    drawn_inter = draw.intersections(img.gray3ch, inter)
    aux.save(img, "intersections", drawn_inter)

    return inter


def magic_lines(img):
    print("finding all lines of board...")
    img = magic_prepare(img)

    got_hough = False
    force = 1.2
    maxgap = 100
    img.slen = (img.bwidth + img.bheigth) * 0.25
    minlen = minlen0 = img.slen
    tvotes = round(minlen / force)
    tangle = np.pi / 360
    h_a = round(np.rad2deg(tangle), 3)

    def _update_magic(force):
        nonlocal minlen, tvotes
        print(f"force: {force=}")
        minlen = minlen0
        tvotes = round(minlen / force)
        return

    incr = 32
    while minlen >= (img.slen/1.5):
        l1 = l2 = ll = 0
        lines = cv2.HoughLinesP(img.canny, 1,
                                tangle, tvotes, None, minlen, maxgap)
        lines = lines[:, 0, :]

        if lines is None:
            minlen = max(img.slen/1.4, minlen - incr)
            tvotes = round(minlen / force)
            continue

        if len(lines) < 18:
            minlen = max(img.slen/1.4, minlen - incr)
            tvotes = round(minlen / force)
            continue

        lines = aux.radius_theta(lines)
        lines = filter_lines(img, lines)
        img.angles = lines_kmeans(img, lines)
        lines = filter_angles(img, lines)
        if len(lines) < 16:
            minlen = max(img.slen/1.4, minlen - incr/2)
            tvotes = round(minlen / force)
            continue

        lines = bundle_lines(lines)
        lines = aux.radius_theta(lines)
        ll = len(lines)
        if ll >= 16:
            dir1, dir2 = split_lines(img, lines)
            l1, l2 = len(dir1), len(dir2)
            if 18 <= ll <= 22 and (9 <= l1 <= 11 and 9 <= l2 <= 11):
                print(f"{ll} # [{l1}][{l2}] ",
                      f"@ {h_a}º,{tvotes},{minlen},{maxgap}")
                got_hough = True
                break

        print(f"{ll} # [{l1}][{l2}] ",
              f"@ {h_a}º,{tvotes},{minlen},{maxgap}")
        minlen -= incr
        tvotes = round(minlen / force)
        if minlen <= (img.slen/1.4):
            if force <= 1.5:
                force += 0.1
                _update_magic(force)
            elif force <= 1.8 and (l1 < 10 or l2 < 10):
                force += 0.1
                _update_magic(force)

    if not got_hough:
        if l1 < 10 or l2 < 10:
            print("magic_lines() failed ",
                  f"@ {180*(tangle/np.pi)},{tvotes},{minlen},{maxgap}")
            exit(1)
        else:
            print("could not find 11 lines in at least one side."
                  "Trying with 10 on both sides.")

    drawn_lines = draw.lines(img, img.gray3ch, dir1, dir2)
    aux.save(img, "hough_magic", drawn_lines)
    return lines


def filter_lines(img, lines):
    if (lines.shape[1] < 6):
        lines = aux.radius_theta(lines)

    rem = np.zeros(lines.shape[0], dtype='uint8')

    for i, line in enumerate(lines):
        x1, y1, x2, y2, r, t = line
        if x1 < (DX+5) and x2 < (DX+5) or y1 < (DX+5) and y2 < (DX+5):
            rem[i] = 1
        elif (img.bwidth - x1) < (DX+5) and (img.bwidth - x2) < (DX+5):
            rem[i] = 1
        elif (img.bheigth - y1) < (DX+5) and (img.bheigth - y2) < (DX+5):
            rem[i] = 1
        elif (x1 < (DX+5) or (img.bwidth - x1) < (DX+5)):
            if (y2 < (DX+5) or (img.bheigth - y2) < (DX+5)):
                rem[i] = 1
        elif (x2 < (DX+5) or (img.bwidth - x2) < (DX+5)):
            if (y1 < (DX+5) or (img.bheigth - y1) < (DX+5)):
                rem[i] = 1
        else:
            rem[i] = 0

    return lines[rem == 0]


def filter_angles(img, lines, tol=15):
    rem = np.zeros(lines.shape[0], dtype='uint8')

    for i, line in enumerate(lines):
        x1, y1, x2, y2, r, t = line
        if abs(t - img.angles[0]) > tol and abs(t - img.angles[1]) > tol:
            if len(img.angles) == 2:
                rem[i] = 1
            elif abs(t - img.angles[2]) > tol:
                rem[i] = 1
            else:
                rem[i] = 0
        else:
            rem[i] = 0

    return lines[rem == 0]


def lines_kmeans(img, lines):
    lines = np.array(lines, dtype='float32')
    if (lines.shape[1] < 6):
        lines = aux.radius_theta(lines)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compact, labels, centers = cv2.kmeans(lines[:, 5], 3, None,
                                          criteria, 10, flags)

    labels = labels.flatten()

    d1 = abs(centers[0] - centers[1])
    d2 = abs(centers[0] - centers[2])
    d3 = abs(centers[1] - centers[2])

    dd1 = d1 < 22.5 and d2 > 22.5 and d3 > 22.5
    dd2 = d2 < 22.5 and d1 > 22.5 and d3 > 22.5
    dd3 = d3 < 22.5 and d1 > 22.5 and d2 > 22.5

    if dd1 or dd2 or dd3:
        compact, labels, centers = cv2.kmeans(lines[:, 5], 2, None,
                                              criteria, 10, flags)

    labels = labels.flatten()

    diff = []
    diff.append((abs(centers[0] - 85), -85))
    diff.append((abs(centers[0] + 85), +85))
    diff.append((abs(centers[1] - 85), -85))
    diff.append((abs(centers[1] + 85), +85))
    if len(centers) > 2:
        diff.append((abs(centers[2] - 85), -85))
        diff.append((abs(centers[2] + 85), +85))

    for d, k in diff:
        if d < 15:
            if len(centers) == 2:
                if abs(centers[0] - k) > 15 and abs(centers[1] - k) > 15:
                    centers = np.append(centers, k)
            else:
                if abs(centers[2] - k) > 15:
                    centers = np.append(centers, k)
            break

    centers = np.round(centers)
    return np.array(centers, dtype='int32')


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

    BR = psum[np.argmax(psum[:, 2])]
    BL = psub[np.argmax(psub[:, 2])]
    TR = psub[np.argmin(psub[:, 2])]
    TL = psum[np.argmin(psum[:, 2])]

    BR = BR[0:2]
    BL = BL[0:2]
    TR = TR[0:2]
    TL = TL[0:2]

    dummy = TR
    TR = BL
    BL = dummy

    BR, BL, TR, TL = broad_corners(img, BR, BL, TR, TL)

    drawn_corners = draw.corners(img, img.gray3ch, BR, BL, TR, TL)
    aux.save(img, "corners", drawn_corners)

    return np.array([BR, BL, TR, TL], dtype='int32')


def perspective_transform(img):
    print("transforming perspective...")
    BR = img.corners[0]
    BL = img.corners[1]
    TR = img.corners[2]
    TL = img.corners[3]
    orig_points = np.array(((TL[0], TL[1]), (TR[0], TR[1]),
                            (BR[0], BR[1]), (BL[0], BL[1])), dtype="float32")

    width = WARP_LEN
    height = WARP_LEN
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
    aux.save(img, "warpclaheG", img.wg)
    aux.save(img, "warpclaheV", img.wv)

    return img


def split_lines(img, lines):
    lines = np.array(lines, dtype='float32')
    if (lines.shape[1] < 6):
        lines = aux.radius_theta(lines)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compact, labels, centers = cv2.kmeans(lines[:, 5], 3, None,
                                          criteria, 10, flags)
    labels = labels.flatten()

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

        labels = labels.flatten()
        A = lines[labels == 0]
        B = lines[labels == 1]

    if len(centers) == 3:
        # redo kmeans using absolute inclination
        lines = aux.radius_theta(lines, abs_angle=True)
        lines = np.array(lines, dtype='float32')
        compact, labels, centers = cv2.kmeans(lines[:, 5], 2, None,
                                              criteria, 10, flags)
        labels = labels.flatten()
        A = lines[labels == 0]
        B = lines[labels == 1]

    return np.int32(A), np.int32(B)


def magic_prepare(img):
    print("preparing image for magic...")
    img = create_cannys(img, w=8.5, saveny=False)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img.canny = cv2.morphologyEx(img.canny, cv2.MORPH_DILATE, kernel)
    aux.save(img, "canny_dilate", img.canny)
    img.canny = cv2.morphologyEx(img.canny, cv2.MORPH_CLOSE, kernel)
    aux.save(img, "canny_closed", img.canny)
    return img


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

    img.bwidth += (DX*2)
    img.bheigth += (DX*2)
    return img
