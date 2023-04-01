import cv2
import numpy as np

import auxiliar as aux
import drawings as draw
import lffilter as lf
import lines as li

from bundle_lines import bundle_lines

WLEN = 640


def find_corners(img):
    img = create_cannys(img)
    vert, hori = magic_lines(img)
    inters = aux.calc_intersections(img.gray3ch, vert, hori)
    canvas = draw.intersections(img.gray3ch, inters)
    aux.save(img, "intersections", canvas)

    img.corners = calc_corners(img, inters)

    intersq = inters.reshape(9, 9, 1, 2)
    intersq = np.flip(intersq, axis=1)
    squares = np.zeros((8, 8, 4, 2), dtype='int32')
    for i in range(0, 8):
        for j in range(0, 8):
            squares[i, j, 0] = intersq[i, j]
            squares[i, j, 1] = intersq[i+1, j]
            squares[i, j, 2] = intersq[i+1, j+1]
            squares[i, j, 3] = intersq[i, j+1]

    canvas = draw.squares(img.board, squares)
    aux.save(img, "A1E4C5H8", canvas)
    squares = np.float32(squares)
    # scale to input size
    squares[:, :, :, 0] /= img.bfact
    squares[:, :, :, 1] /= img.bfact
    # position board bounding box
    squares[:, :, :, 0] += img.x0
    squares[:, :, :, 1] += img.y0

    img.squares = np.array(np.round(squares), dtype='int32')
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
    maxgap = round(minlen0 / 8)
    tvotes = round(minlen0 / 1.2)
    angle = 0.5  # degrees
    tangle = np.deg2rad(angle)  # radians

    gotmin = False
    minlen = 0
    for minlen in range(minlen0, round(minlen0 / 1.6), -16):
        tvotes = round(minlen / 1.2)
        print(f"trying @{angle}º, {tvotes}, {minlen}, {maxgap}")
        lines = cv2.HoughLinesP(img.canny, 1,
                                tangle, tvotes, None, minlen, maxgap)
        lines = lines[:, 0, :]
        lines = bundle_lines(lines)
        if lines is not None and len(lines) >= 12:
            gotmin = True
            break
    if not gotmin:
        print("magic_lines() failed @ {angle}º, {tvotes}, {minlen}, {maxgap}")
        # aux.save(img, "lastcanny", img.canny)
        canvas = draw.lines(img.gray3ch, lines)
        exit(1)

    canvas = draw.lines(img.gray3ch, lines)
    # aux.save(img, "hough_magic000", canvas)

    lines, _ = aux.radius_theta(lines)
    minlen0 = minlen = min(np.mean(lines[:, 4]), 300)
    maxgap = round(minlen0 / 4)
    tvotes = round(minlen0 * 1)
    print(f"{minlen=}")
    ll = lv = lh = 0
    while (lv < 9 or lh < 9) and tvotes > (minlen0 / 1.6):
        minlen = max(minlen - 5, minlen0 / 1.2)
        tvotes -= 10
        lines = cv2.HoughLinesP(img.canny, 1,
                                tangle, tvotes, None, minlen, maxgap)
        if (ll := len(lines)) < 16:
            print(f"{ll} @ {angle}, {tvotes}, {minlen}, {maxgap}")
            continue
        lines = lines[:, 0, :]
        lines = bundle_lines(lines)
        lines, _ = aux.radius_theta(lines)
        vert, hori = li.split_lines(lines)
        vert, hori = li.filter_byangle(vert, hori)
        vert, hori = li.sort_lines(vert, hori)
        lv, lh = len(vert), len(hori)
        ll = lv + lh
        print(f"{ll} # [{lv}][{lh}] @",
              f"{angle}º, {tvotes}, {minlen}, {maxgap}")

    vert, hori, extras = li.add_outer_wrap(img, vert, hori)
    # vert, hori = li.add_middle(vert, hori)
    vert, hori = aux.radius_theta(vert, hori)
    vert, hori = li.sort_lines(vert, hori)
    vert, hori = li.remove_extras2(vert, hori, img.bwidth, img.bheigth)
    # vert, hori = li.add_last_outer(vert, hori, medv, medh)
    if True or len(vert) != 9 or len(hori) != 9:
        aux.save(img, "len(vert)!=9 or len(hori)!=9", img.canny)

    canvas = draw.lines(img.gray3ch, vert, hori, extras)
    aux.save(img, "hough_magic", canvas)
    return vert, hori


def calc_corners(img, inters):
    inter = np.copy(inters)
    print("calculating 4 corners of board...")
    inter = inter.reshape((-1, 2))
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
