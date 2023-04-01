import cv2
import numpy as np

import auxiliar as aux
import drawings as draw
import lffilter as lf
import lines as li

from bundle_lines import bundle_lines

WLEN = 640


def find_squares(img):
    print("generating 3 channel gray warp image for drawings...")
    img.warp3ch = cv2.cvtColor(img.wg, cv2.COLOR_GRAY2BGR)
    img = create_wcannys(img)
    vert, hori = w_lines(img)
    vert, hori = magic_vert_hori(img, vert, hori)

    inter = aux.calc_intersections(img.warp3ch, vert, hori)
    inter = inter.reshape((-1, 2))
    canvas = draw.intersections(img.warp3ch, [inter])
    aux.save(img, "intersections", canvas)
    if len(inter) != 81:
        print("There should be exacly 81 intersections")
        exit(1)
    squares = calc_squares(img, inter)

    print("transforming squares corners to original coordinate system...")
    print(f"{squares=}")
    print(f"{squares.shape=}")
    sqback = np.zeros(squares.shape, dtype='float32')
    print(f"{sqback=}")
    print(f"{sqback.shape=}")
    for i in range(0, 8):
        sqback[i] = cv2.perspectiveTransform(squares[i], img.warpInvMatrix)
    squares = np.round(sqback)
    squares = np.array(sqback, dtype='int32')

    canvas = draw.squares(img.board, squares)
    aux.save(img, "A1E4C5H8", canvas)

    # scale to input size
    sqback[:, :, :, 0] /= img.bfact
    sqback[:, :, :, 1] /= img.bfact
    # position board bounding box
    sqback[:, :, :, 0] += img.x0
    sqback[:, :, :, 1] += img.y0

    img.squares = np.array(np.round(sqback), dtype='int32')
    canvas = draw.squares(img.BGR, img.squares)
    # aux.save(img, "A1E4C5H8", canvas)

    return img


def create_wcannys(img):
    print("finding edges for gray, V warp images...")
    cannyG = aux.find_edges(img, img.wg, lowpass=lf.ffilter)
    cannyV = aux.find_edges(img, img.wv, lowpass=lf.ffilter)
    # aux.save(img, "wcannyG", cannyG)
    # aux.save(img, "wcannyV", cannyV)
    img.wcanny = cv2.bitwise_or(cannyG, cannyV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img.wcanny = cv2.morphologyEx(img.wcanny, cv2.MORPH_DILATE, kernel)
    # aux.save(img, "wcanny_dilate", img.wcanny)
    img.wcanny = cv2.morphologyEx(img.wcanny, cv2.MORPH_CLOSE, kernel)
    # aux.save(img, "wcanny_close", img.wcanny)
    return img


def w_lines(img):
    print("finding vertical and horizontal lines...")
    got_hough = False
    force = 1.2
    maxgap = img.wwidth / 8
    minlen = minlen0 = img.wwidth * 0.8
    tvotes = round(minlen / force)
    tangle = np.pi / 360
    h_a = round(np.rad2deg(tangle), 2)

    def _update_magic(force):
        nonlocal minlen, tvotes
        print(f"{force=}")
        minlen = minlen0
        tvotes = round(minlen / force)
        return

    incr = 32
    while minlen >= (minlen0/1.5):
        l1 = l2 = ll = 0
        lines = cv2.HoughLinesP(img.wcanny, 1,
                                tangle, tvotes, None, minlen, maxgap)
        lines = lines[:, 0, :]

        if lines is None or len(lines) < 18:
            minlen = max(minlen0/1.4, minlen - incr)
            tvotes = round(minlen / force)
            continue

        lines, _ = aux.radius_theta(lines)
        lines = filter_90(lines)
        ll = len(lines)
        if ll < 10:
            minlen = max(minlen0/1.4, minlen - incr/2)
            tvotes = round(minlen / force)
            continue

        lines = bundle_lines(lines)
        lines, _ = aux.radius_theta(lines)
        vert, hori = aux.geo_lines(lines)
        l1, l2 = len(vert), len(hori)
        if 18 <= ll and (9 <= l1 <= 11 and 9 <= l2 <= 11):
            print(f"{ll}>{len(lines)} # [{l1}][{l2}] ",
                  f"@ {h_a}º,{tvotes},{minlen},{maxgap}")
            got_hough = True
            break

        print(f"{ll} # [{l1}][{l2}] ",
              f"@ {h_a}º,{tvotes},{minlen},{maxgap}")
        minlen -= incr
        tvotes = round(minlen / force)
        if minlen <= (minlen0/1.4):
            force += 0.1
            _update_magic(force)
        if tvotes < 200:
            break

    if not got_hough:
        if l1 < 6 and l2 < 6:
            print("magic_lines() failed:",
                  f"{ll} # [{l1}][{l2}]",
                  f"@ {h_a}º,{tvotes},{minlen},{maxgap}")
            exit(1)

    canvas = draw.lines(img.warp3ch, vert, hori)
    # aux.save(img, "wmagic", canvas)
    return vert, hori


def filter_90(lines):
    rem = np.zeros(lines.shape[0], dtype='uint8')

    for i, t in enumerate(lines[:, 5]):
        if abs(t - 90) > 4 and abs(t + 90) > 4 and abs(t) > 4:
            rem[i] = 1
        else:
            rem[i] = 0

    return lines[rem == 0]


def magic_vert_hori(img, vert, hori):
    canvas = draw.lines(img.warp3ch, vert, hori)
    # aux.save(img, "verthori0", canvas)
    print("adjusting vertical and horizontal lines...")
    lv, lh = len(vert), len(hori)
    if lv <= 5 and lh <= 5 or (lh < 1 > lv):
        print("There should be at least 6 lines of one direction",
              "and 1 line on the other")
        exit(1)

    def _check_save(title):
        nonlocal lv, lh, vert, hori
        if lv != len(vert) or lh != len(hori):
            canvas = draw.lines(img.warp3ch, vert, hori)
            # aux.save(img, title, canvas)
            lv, lh = len(vert), len(hori)
        return

    print("calculating median distances...")
    distv, disth = li.get_distances(vert, hori)
    medv, medh = li.mean_dist(distv, disth)
    print(f"{medv=}")
    print(f"{medh=}")

    if lv >= 5:
        print("removing for sure wrong vertical lines...")
        vert = li.wrong_lines(vert, distv, medv, tol=4)
    if lh >= 5:
        print("removing for sure wrong horizontal lines...")
        hori = li.wrong_lines(hori, disth, medh, tol=4)
    _check_save("rem_wrong")

    if lv >= 5 and lh >= 5:
        print("updating median distances...")
        distv, disth = li.get_distances(vert, hori)
        medv, medh = li.mean_dist(distv, disth)
        print("chosing best lines...")
        vert = li.right_lines(vert, distv, medv)
        hori = li.right_lines(hori, disth, medh)
        _check_save("right_lines")

    vert, hori = li.add_wouter(vert, hori, medv, medh)
    _check_save("add_wouter")
    vert, hori = li.add_middle(vert, hori)
    _check_save("add_middle")
    vert, hori = li.remove_extras(vert, hori, img.wwidth, img.wheigth)
    _check_save("rem_extras")
    vert, hori = li.add_last_outer(vert, hori, medv, medh)
    _check_save("last_outer")

    if len(vert) != 9 or len(hori) != 9:
        print("There should be exactly 9 vertical and 9 horizontal lines")
        exit(1)
    return vert, hori


def calc_squares(img, inter):
    print("calculating squares corners...")
    inter = inter[np.argsort(inter[:, 0])]
    intersq = np.zeros((9, 9, 2), dtype='int32')
    interA = inter[0:9]   # A
    interB = inter[9:18]   # B
    interC = inter[18:27]  # C
    interD = inter[27:36]  # D
    interE = inter[36:45]  # E
    interF = inter[45:54]  # F
    interG = inter[54:63]  # G
    interH = inter[63:72]  # H
    interZ = inter[72:81]  # right

    intersq[0, :] = interA[np.argsort(interA[:, 1])[::-1]]  # A
    intersq[1, :] = interB[np.argsort(interB[:, 1])[::-1]]  # B
    intersq[2, :] = interC[np.argsort(interC[:, 1])[::-1]]  # C
    intersq[3, :] = interD[np.argsort(interD[:, 1])[::-1]]  # D
    intersq[4, :] = interE[np.argsort(interE[:, 1])[::-1]]  # E
    intersq[5, :] = interF[np.argsort(interF[:, 1])[::-1]]  # F
    intersq[6, :] = interG[np.argsort(interG[:, 1])[::-1]]  # G
    intersq[7, :] = interH[np.argsort(interH[:, 1])[::-1]]  # H
    intersq[8, :] = interZ[np.argsort(interZ[:, 1])[::-1]]  # right

    squares = np.zeros((8, 8, 4, 2), dtype='int32')
    for i in range(0, 8):
        for j in range(0, 8):
            squares[i, j, 0] = intersq[i, j]
            squares[i, j, 1] = intersq[i+1, j]
            squares[i, j, 2] = intersq[i+1, j+1]
            squares[i, j, 3] = intersq[i, j+1]

    return np.array(squares, dtype='float32')
