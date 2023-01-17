import cv2
import numpy as np

import auxiliar as aux
import drawings as draw
import lffilter as lf
import lines as li

from bundle_lines import bundle_lines

MINLEN = 240
bonus = 0


def find_squares(img):
    img = create_cannys(img)
    vert, hori = magic_lines(img)
    inters = aux.calc_intersections(img.gray3ch, vert, hori)
    canvas = draw.points(img.gray3ch, inters)
    # aux.save(img, "intersections", canvas)

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
    # aux.save(img, "A1E4C5H8", canvas)
    squares = np.float32(squares)
    # scale to input size
    squares[:, :, :, 0] /= img.bfact
    squares[:, :, :, 1] /= img.bfact
    # position board bounding box
    squares[:, :, :, 0] += img.x0
    squares[:, :, :, 1] += img.y0

    img.squares = np.array(np.round(squares), dtype='int32')
    return img


def create_cannys(img, bonus=0):
    print("finding edges for gray, S, V images...")
    cannyG, got_canny = aux.find_edges(img, img.G,
                                       lowpass=lf.ffilter, bonus=bonus)
    if not got_canny:
        aux.save(img, "cannyG", cannyG)
    cannyV, got_canny = aux.find_edges(img, img.V,
                                       lowpass=lf.ffilter, bonus=bonus)
    if not got_canny:
        aux.save(img, "cannyV", cannyV)
    img.canny = cv2.bitwise_or(cannyG, cannyV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img.canny = cv2.morphologyEx(img.canny, cv2.MORPH_DILATE, kernel)
    img.canny = cv2.morphologyEx(img.canny, cv2.MORPH_CLOSE, kernel)
    return img


def magic_lines(img):
    print("finding all lines of board...")
    global bonus

    angle = 0.5  # degrees
    tangle = np.deg2rad(angle)  # radians
    minlen0 = minlen = MINLEN
    maxgap = round(minlen0 / 4)
    tvotes = round(minlen0 * 1)
    ll = lv = lh = 0
    while (lv < 9 or lh < 9) and tvotes > (minlen0 / 1.4):
        minlen = max(minlen - 8, minlen0 / 1.2)
        tvotes -= 12
        lines = cv2.HoughLinesP(img.canny, 1,
                                tangle, tvotes, None, minlen, maxgap)
        if (ll := lines.shape[0]) < 16:
            print(f"{ll} @ {angle}, {tvotes}, {minlen}, {maxgap}")
            continue
        lines = lines[:, 0, :]
        lines = bundle_lines(img, lines)
        lines, _ = aux.radius_theta(lines)
        vert, hori = li.split_lines(lines)
        vert, hori = li.filter_byangle(vert, hori)
        vert, hori = li.sort_lines(vert, hori)
        lv, lh = vert.shape[0], hori.shape[0]
        ll = lv + lh
        print(f"{ll} # [{lv}][{lh}] @",
              f"{angle}º, {tvotes}, {minlen}, {maxgap}")
    if (lv < 8 or lh < 8) and bonus < 3:
        bonus += 1
        print(f"{bonus=}")
        img = create_cannys(img, bonus=bonus)
        vert, hori = magic_lines(img)
        return vert, hori

    if lv < 9 or lh < 9:
        aux.save(img, f"canny{lv=}_{lh=}", img.canny)
    vert, hori = li.shorten_byinter(img, img.bwidth, img.bheigth, vert, hori)
    vert, hori = li.add_outer_wrap(img, vert, hori)
    vert, hori = li.sort_lines(vert, hori)
    vert, hori = li.shorten_byinter(img, img.bwidth, img.bheigth, vert, hori)
    vert, hori = li.remove_extras(vert, hori, img.bwidth, img.bheigth)
    vert, hori = li.add_middle(vert, hori)
    vert, hori = li.sort_lines(vert, hori)
    vert, hori = li.remove_extras(vert, hori, img.bwidth, img.bheigth)

    canvas = draw.lines(img.gray3ch, vert, hori)
    aux.save(img, "hough_magic", canvas)
    return vert, hori
