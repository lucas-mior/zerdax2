import cv2
import numpy as np

import auxiliar as aux
import drawings as draw

from bundle_lines import bundle_lines

WLEN = 640
DX = 40


def find_squares(img):
    img = pre_process(img)
    img = create_wcannys(img, w=10)
    vert, hori = w_lines(img)
    vert, hori = magic_vert_hori(img, vert, hori)

    inter = aux.calc_intersections(img, img.warp3ch, vert, hori)
    if len(inter) != 81:
        print("There should be exacly 81 intersections")
        exit(1)
    squares = calc_squares(img, inter)

    print("transforming squares corners to original coordinate system...")
    sqback = np.zeros(squares.shape, dtype='float32')
    for i in range(0, 8):
        sqback[i] = cv2.perspectiveTransform(squares[i], img.warpInvMatrix)
    img.sqback = np.array(np.round(sqback), dtype='int32')

    canvas = draw.squares(img.board, img.sqback)
    # aux.save(img, "A1E4C5H8", canvas)

    # remove black border
    sqback[:, :, :, 0] -= DX
    sqback[:, :, :, 1] -= DX
    # scale to input size
    sqback[:, :, :, 0] /= img.bfact
    sqback[:, :, :, 1] /= img.bfact
    # position board bounding box
    sqback[:, :, :, 0] += img.x0
    sqback[:, :, :, 1] += img.y0

    img.sqback = np.array(np.round(sqback), dtype='int32')
    canvas = draw.squares(img.BGR, img.sqback)
    aux.save(img, "A1E4C5H8", canvas)

    return img


def create_wcannys(img, w=10, thighg=200, thighv=200):
    print("finding edges for gray, V warp images...")
    cannyG = aux.find_canny(img.wg, wmin=w, thigh=thighg)
    cannyV = aux.find_canny(img.wv, wmin=w, thigh=thighv)
    # aux.save(img, "wcannyG", cannyG)
    # aux.save(img, "wcannyV", cannyV)
    img.wcanny = cv2.bitwise_or(cannyG, cannyV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img.wcanny = cv2.morphologyEx(img.wcanny, cv2.MORPH_CLOSE, kernel)

    img.wcanny = cv2.morphologyEx(img.wcanny, cv2.MORPH_DILATE, kernel)
    img.wcanny = cv2.morphologyEx(img.wcanny, cv2.MORPH_CLOSE, kernel)
    # aux.save(img, "wcanny", img.wcanny)
    return img


def w_lines(img):
    print("finding vertical and horizontal lines...")
    got_hough = False
    force = 1.2
    maxgap = img.wwidth / 4
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

        lines = aux.radius_theta(lines)
        lines = filter_90(lines)
        ll = len(lines)
        if ll < 10:
            minlen = max(img.slen/1.4, minlen - incr/2)
            tvotes = round(minlen / force)
            continue

        lines = bundle_lines(lines)
        lines = aux.radius_theta(lines)
        vert, hori = aux.geo_lines(lines)
        l1, l2 = len(vert), len(hori)
        if 18 <= ll and (9 <= l1 <= 11 and 9 <= l2 <= 11):
            print(f"{ll}>{len(vert)} # [{l1}][{l2}] ",
                  f"@ {h_a}º,{tvotes},{minlen},{maxgap}")
            got_hough = True
            break

        print(f"{ll} # [{l1}][{l2}] ",
              f"@ {h_a}º,{tvotes},{minlen},{maxgap}")
        minlen -= incr
        tvotes = round(minlen / force)
        if minlen <= (img.slen/1.4):
            force += 0.1
            _update_magic(force)

    if not got_hough:
        if l1 < 7 or l2 < 7:
            print("magic_lines() failed:",
                  f"{ll} # [{l1}][{l2}]",
                  f"@ {h_a}º,{tvotes},{minlen},{maxgap}")
            # exit(1)

    canvas = draw.lines(img.warp3ch, vert, hori)
    aux.save(img, "wmagic", canvas)
    return vert, hori


def filter_90(lines):
    rem = np.zeros(lines.shape[0], dtype='uint8')

    for i, t in enumerate(lines[:, 5]):
        if abs(t - 90) > 4 and abs(t + 90) > 4 and abs(t) > 4:
            rem[i] = 1
        else:
            rem[i] = 0

    return lines[rem == 0]


def get_distances(vert, hori):
    def _get_dist(lines, kind):
        dist = np.zeros((lines.shape[0], 2), dtype='int32')
        x1 = (lines[1, kind] + lines[1, kind+2])/2
        x2 = (lines[0, kind] + lines[0, kind+2])/2
        dist[0, 0] = abs(x1 - x2)
        dist[0, 1] = abs(x1 - x2)
        i = 0
        for i in range(1, len(lines) - 1):
            x1 = (lines[i-1, kind]+lines[i-1, kind+2])/2
            x2 = (lines[i+0, kind]+lines[i+0, kind+2])/2
            x3 = (lines[i+1, kind]+lines[i+1, kind+2])/2
            dist[i, 0] = abs(x1 - x2)
            dist[i, 1] = abs(x2 - x3)
        i += 1
        x1 = (lines[i-1, kind]+lines[i-1, kind+2])/2
        x2 = (lines[i+0, kind]+lines[i+0, kind+2])/2
        dist[i, 0] = abs(x1 - x2)
        dist[i, 1] = abs(x1 - x2)
        return dist

    distv = _get_dist(vert, 0)
    disth = _get_dist(hori, 1)
    return distv, disth


def mean_dist(distv, disth):
    def _mean_dist(dist):
        med1 = np.median(dist[:, 0])
        med2 = np.median(dist[:, 1])
        med = round((med1 + med2)/2)
        return med

    medv = _mean_dist(distv)
    medh = _mean_dist(disth)
    return medv, medh


def wrong_lines(dist, med):
    rem = np.zeros(dist.shape[0], dtype='uint8')

    for i, d in enumerate(dist):
        if abs(d[0] - med) > 8 and abs(d[1] - med) > 8:
            rem[i] = 1
        else:
            rem[i] = 0
    return rem


def right_lines(dist, med):
    cer = np.zeros(dist.shape[0], dtype='uint8')

    for i, d in enumerate(dist):
        if abs(d[0] - med) < 8 and abs(d[1] - med) < 8:
            cer[i] = 1
        else:
            cer[i] = 0
    return cer


def magic_vert_hori(img, vert, hori):
    canvas = draw.lines(img.warp3ch, vert, hori)
    aux.save(img, "verthori0", canvas)
    print("adjusting vertical and horizontal lines...")
    lv, lh = len(vert), len(hori)

    def _check_save(title):
        nonlocal lv, lh, vert, hori
        if lv != len(vert) or lh != len(hori):
            canvas = draw.lines(img.warp3ch, vert, hori)
            aux.save(img, title, canvas)
            lv, lh = len(vert), len(hori)
        return

    print("calculating median distances...")
    distv, disth = get_distances(vert, hori)
    medv, medh = mean_dist(distv, disth)
    print(f"{medv=}")
    print(f"{medh=}")

    print("removing for sure wrong lines...")
    remv = wrong_lines(distv, medv)
    remh = wrong_lines(disth, medh)
    vert = vert[remv == 0]
    hori = hori[remh == 0]
    _check_save("rem_wrong")

    print("updating median distances...")
    distv, disth = get_distances(vert, hori)
    medv, medh = mean_dist(distv, disth)

    print("chosing best lines...")
    cerv = right_lines(distv, medv)
    cerh = right_lines(disth, medh)
    vert = vert[cerv == 1]
    hori = hori[cerh == 1]
    _check_save("right_lines")

    vert, hori = add_outer(vert, hori, medv, medh)
    _check_save("add_outer")
    vert, hori = add_middle(vert, hori, medv, medh)
    _check_save("add_middle")
    vert, hori = remove_extras(vert, hori)
    _check_save("rem_extras")
    vert, hori = add_last_outer(vert, hori, medv, medh)
    _check_save("last_outer")

    if len(vert) != 9 or len(hori) != 9:
        print("There should be exactly 9 vertical and 9 horizontal lines")
        exit(1)
    return vert, hori


def add_outer(vert, hori, medv, medh):
    print("adding missing outer lines...")
    while abs(vert[0, 0] - 0) > (medv + 5):
        x1 = vert[0, 0] - medv
        y1 = vert[0, 1]
        x2 = vert[0, 2] - medv
        y2 = vert[0, 3]
        new = np.array([[x1, y1, x2, y2, 0, 0]], dtype='int32')
        vert = np.append(vert, new, axis=0)
        vert = vert[np.argsort(vert[:, 0])]
    while abs(vert[-1, 0] - WLEN) > (medv + 5):
        x1 = vert[-1, 0] + medv
        y1 = vert[-1, 1]
        x2 = vert[-1, 2] + medv
        y2 = vert[-1, 3]
        new = np.array([[x1, y1, x2, y2, 0, 0]], dtype='int32')
        vert = np.append(vert, new, axis=0)
        vert = vert[np.argsort(vert[:, 0])]
    while abs(hori[0, 1] - 0) > (medh + 5):
        x1 = hori[0, 0]
        y1 = hori[0, 1] - medh
        x2 = hori[0, 2]
        y2 = hori[0, 3] - medh
        new = np.array([[x1, y1, x2, y2, 0, 0]], dtype='int32')
        hori = np.append(hori, new, axis=0)
        hori = hori[np.argsort(hori[:, 1])]
    while abs(hori[-1, 1] - WLEN) > (medh + 5):
        x1 = hori[-1, 0]
        y1 = hori[-1, 1] + medh
        x2 = hori[-1, 2]
        y2 = hori[-1, 3] + medh
        new = np.array([[x1, y1, x2, y2, 0, 0]], dtype='int32')
        hori = np.append(hori, new, axis=0)
        hori = hori[np.argsort(hori[:, 1])]

    return vert, hori


def add_middle(vert, hori, medv, medh):
    print("adding missing middle lines...")

    def _add_middle(lines, med, kind):
        i = 0
        while i < (len(lines) - 1):
            if abs(lines[i, kind] - lines[i+1, kind]) > (med*1.5):
                if kind == 0:
                    x1 = lines[i, 0] + med
                    y1 = lines[i, 1]
                    x2 = lines[i, 2] + med
                    y2 = lines[i, 3]
                else:
                    x1 = lines[i, 0]
                    y1 = lines[i, 1] + med
                    x2 = lines[i, 2]
                    y2 = lines[i, 3] + med
                new = np.array([[x1, y1, x2, y2, 0, 0]], dtype='int32')
                lines = np.append(lines, new, axis=0)
                lines = lines[np.argsort(lines[:, kind])]
            i += 1
        return lines

    vert = _add_middle(vert, medv, 0)
    hori = _add_middle(hori, medh, 1)
    return vert, hori


def remove_extras(vert, hori):
    print("removing extra outer lines...")
    v = len(vert)
    h = len(hori)
    if v <= 9 and h <= 9:
        return vert, hori

    def _rem_extras(lines, num, kind):
        if num == 10:
            d1 = abs(lines[0, kind] - 0)
            d2 = abs(lines[-1, kind] - WLEN)
            if d1 < d2:
                lines = lines[1:]
            else:
                lines = lines[:-1]
        elif num == 11:
            lines = lines[1:-1]
        elif num >= 12:
            print("There are 12 or more vertical lines")
            lines = lines[1:-1]
            lines = _rem_extras(lines, len(lines), kind)
        return lines

    vert = _rem_extras(vert, v, 0)
    hori = _rem_extras(hori, h, 1)
    return vert, hori


def add_last_outer(vert, hori, medv, medh):
    print("adding last missing outer lines...")
    v = len(vert)
    h = len(hori)
    if v == 9 and h == 9:
        return vert, hori

    def _add_last_outer(lines, num, med, kind):
        if num < 8:
            print("7 or less lines, there should be at least 8")
        elif num == 8:
            d1 = abs(lines[0, 0] - 0)
            d2 = abs(lines[-1, 0] - WLEN)
            if d1 > d2:
                if d1 >= med:
                    x1 = lines[0, 0] - med
                    y1 = lines[0, 1]
                    x2 = lines[0, 2] - med
                    y2 = lines[0, 3]
                    new = np.array([[x1, y1, x2, y2, 0, 0]], dtype='int32')
                else:
                    print("not enough space for inserting missing 9th line")
                    exit(1)
            else:
                if d2 >= med:
                    x1 = lines[-1, 0] + med
                    y1 = lines[-1, 1]
                    x2 = lines[-1, 2] + med
                    y2 = lines[-1, 3]
                    new = np.array([[x1, y1, x2, y2, 0, 0]], dtype='int32')
                else:
                    print("not enough space for inserting missing 9th line")
                    exit(1)

            lines = np.append(lines, new, axis=0)
            lines = lines[np.argsort(lines[:, kind])]
        return lines

    vert = _add_last_outer(vert, v, medv, 0)
    hori = _add_last_outer(hori, h, medh, 1)
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


def pre_process(img):
    print("generating 3 channel gray warp image for drawings...")
    img.warp3ch = cv2.cvtColor(img.wg, cv2.COLOR_GRAY2BGR)

    print("applying gaussian blur...")
    img.wg = cv2.GaussianBlur(img.wg, (5, 5), 0.5)
    img.wv = cv2.GaussianBlur(img.wv, (5, 5), 0.5)
    # aux.save(img, "wGblur", img.wg)
    # aux.save(img, "wVblur", img.wv)

    # print("filtering warp image...")
    # img.wg = lf.ffilter(img.wg)
    # img.wv = lf.ffilter(img.wv)
    # # aux.save(img, "wg_filter", img.wg)
    # # aux.save(img, "wv_filter", img.wv)
    return img
