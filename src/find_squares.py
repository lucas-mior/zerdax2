import cv2
import numpy as np

import auxiliar as aux

from bundle_lines import bundle_lines
import lffilter as lf
import random

WARPED_LEN = 640
DX = 40


def draw_squares(img, image):
    canvas = np.zeros(image.shape, dtype='uint8')
    cv2.drawContours(canvas, [img.sqback[0, 0]], -1,  # A1
                     color=(255, 0, 0), thickness=img.thick)
    cv2.drawContours(canvas, [img.sqback[4, 3]], -1,  # E4
                     color=(0, 255, 0), thickness=img.thick)
    cv2.drawContours(canvas, [img.sqback[2, 4]], -1,  # C5
                     color=(0, 0, 255), thickness=img.thick)
    cv2.drawContours(canvas, [img.sqback[7, 7]], -1,  # H8
                     color=(0, 220, 220), thickness=img.thick)
    cv2.addWeighted(image, 0.6, canvas, 0.4, 0, canvas)
    return canvas


def find_squares(img):
    print("generating 3 channel gray warp image for drawings...")
    img.warp3ch = cv2.cvtColor(img.wg, cv2.COLOR_GRAY2BGR)
    print("filtering warp image...")
    img.wg = lf.ffilter(img.wg)
    img.wv = lf.ffilter(img.wv)

    vert, hori = w_lines(img)
    # aux.save_lines(img, "verthori0", vert, hori)
    vert, hori = magic_vert_hori(img, vert, hori)

    inter = calc_intersections(img, vert, hori)
    squares = calc_squares(img, inter)
    squares = np.float32(squares)

    print("transforming squares corners to original coordinate system...")
    sqback = np.zeros(squares.shape, dtype='float32')
    for i in range(0, 8):
        sqback[i] = cv2.perspectiveTransform(squares[i], img.warpInvMatrix)
    img.sqback = np.int32(np.round(sqback))

    squares_drawn = draw_squares(img, img.board)
    # aux.save(img, "A1E4C5", squares_drawn)

    # remove black border
    sqback[:, :, :, 0] -= DX
    sqback[:, :, :, 1] -= DX
    # scale to input size
    sqback[:, :, :, 0] /= img.bfact
    sqback[:, :, :, 1] /= img.bfact
    # position board bounding box
    sqback[:, :, :, 0] += img.x0
    sqback[:, :, :, 1] += img.y0

    img.sqback = np.int32(np.round(sqback))
    squares_drawn = draw_squares(img, img.BGR)
    aux.save(img, "A1E4C5", squares_drawn)

    return img


def create_wcannys(img, w=10, thighg=220, thighv=220):
    print("finding edges for gray, V warp images...")
    cannyG = aux.find_canny(img.wg, wmin=w, thigh=thighg)
    cannyV = aux.find_canny(img.wv, wmin=w, thigh=thighv)
    img.wcanny = cv2.bitwise_or(cannyG, cannyV)

    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img.wcanny = cv2.morphologyEx(img.wcanny, cv2.MORPH_CLOSE, k_close)
    return img


def w_lines(img):
    print("finding vertical and horizontal lines...")
    img = create_wcannys(img, w=10)
    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img.wcanny = cv2.morphologyEx(img.wcanny, cv2.MORPH_DILATE, k_dil)
    img.wcanny = cv2.morphologyEx(img.wcanny, cv2.MORPH_CLOSE, k_dil)
    # aux.save(img, "wcanny", img.wcanny)
    got_hough = False

    def _update_wlines(force, le):
        nonlocal passed, tangle, minlen, tvotes
        passed += 1
        print("passed:", passed)
        tangle = tangle0
        minlen = round((img.wwidth) * le)
        tvotes = round(minlen0 / force)

    tangle = tangle0 = np.pi / 720
    minlen = minlen0 = round((img.wwidth)*0.8)
    tvotes = round(minlen0 / 2)
    maxgap = 60
    passed = 0
    newgap = False
    vert = hori = None
    minlines = 9
    while tangle <= (np.pi / 350):
        lv = lh = 0
        th = 180*(tangle/np.pi)
        lines = cv2.HoughLinesP(img.wcanny, 1,
                                tangle, tvotes, None, minlen, maxgap)
        if lines is not None:
            lines = aux.radius_theta(lines)
            lines = filter_90(lines)
            if len(lines) > 16:
                lines = bundle_lines(lines)
                lines = aux.radius_theta(lines)
                vert, hori = aux.geo_lines(lines)
                lv = len(vert)
                lh = len(hori)
                if lv >= minlines and lh >= minlines:
                    print(f"{len(lines)} lines [{lv}][{lh}] ",
                          f"@ {th:1=.3f}º, {tvotes}, {minlen}, {maxgap}")
                    got_hough = True
                    break
            if th > random.uniform(0, th*1):
                print(f"{len(lines)} lines [{lv}][{lh}] ",
                      f"@ {th:1=.3f}º, {tvotes}, {minlen}, {maxgap}")
        tangle += np.pi / 3600
        if tangle >= (np.pi / 360):
            if passed == 0:
                _update_wlines(2.2, 0.75)
            elif passed == 1:
                _update_wlines(2.3, 0.70)
            elif passed == 2:
                _update_wlines(2.3, 0.65)
            elif passed == 3 and (lv < 8 or lh < 8) and not newgap:
                _update_wlines(2, 0.80)
                maxgap = 80
                newgap = True
                passed = 0

    if not got_hough:
        # aux.save(img, "lastcanny", img.wcanny)
        # aux.save_lines(img, "lastverthori0", vert[:, 0, :], hori[:, 0, :])
        if lv < 8 or lh < 8:
            print(f"FAILED @ {180*(tangle/np.pi)},{tvotes},{minlen},{maxgap}")
            exit(1)
        else:
            print("failed to find at least 9 lines, trying with 8")

    return vert[:, 0, :], hori[:, 0, :]


def filter_90(lines):
    rem = np.zeros(lines.shape[0], dtype='uint8')

    for i, line in enumerate(lines):
        for x1, y1, x2, y2, r, t in line:
            if abs(t - 90) > 4 and abs(t + 90) > 4 and abs(t) > 4:
                rem[i] = 1
            else:
                rem[i] = 0
        i += 1

    A = lines[rem == 0]
    lines = A
    return lines


def get_distances(vert, hori):
    distv = np.zeros((vert.shape[0], 2), dtype='int32')
    x1 = (vert[1, 0] + vert[1, 2])/2
    x2 = (vert[0, 0] + vert[0, 2])/2
    distv[0, 0] = abs(x1 - x2)
    distv[0, 1] = abs(x1 - x2)
    i = 0
    for i in range(1, len(vert) - 1):
        x1 = (vert[i-1, 0]+vert[i-1, 2])/2
        x2 = (vert[i+0, 0]+vert[i+0, 2])/2
        x3 = (vert[i+1, 0]+vert[i+1, 2])/2
        distv[i, 0] = abs(x1 - x2)
        distv[i, 1] = abs(x2 - x3)
    i += 1
    x1 = (vert[i-1, 0]+vert[i-1, 2])/2
    x2 = (vert[i+0, 0]+vert[i+0, 2])/2
    distv[i, 0] = abs(x1 - x2)
    distv[i, 1] = abs(x1 - x2)

    disth = np.zeros((hori.shape[0], 2), dtype='int32')
    x1 = (hori[1, 1]+hori[1, 3])/2
    x2 = (hori[0, 1]+hori[0, 3])/2
    disth[0, 0] = abs(x1 - x2)
    disth[0, 1] = abs(x1 - x2)

    i = 0
    for i in range(1, len(hori)-1):
        x1 = (hori[i-1, 1]+hori[i-1, 3])/2
        x2 = (hori[i+0, 1]+hori[i+0, 3])/2
        x3 = (hori[i+1, 1]+hori[i+1, 3])/2
        disth[i, 0] = abs(x1 - x2)
        disth[i, 1] = abs(x2 - x3)
    i += 1
    x1 = (hori[i-1, 1]+hori[i-1, 3])/2
    x2 = (hori[i+0, 1]+hori[i+0, 3])/2
    disth[i, 0] = abs(x1 - x2)
    disth[i, 1] = abs(x1 - x2)

    return distv, disth


def calc_intersections(img, vert, hori):
    print("calculating intersections...")
    inter = []
    last = (0, 0)

    i = 0
    for x1, y1, x2, y2, r, t in vert:
        j = 0
        for xx1, yy1, xx2, yy2, rr, tt in hori:
            if (x1, y1) == (xx1, yy1) and (x2, y2) == (xx2, yy2):
                continue

            xdiff = (x1 - x2, xx1 - xx2)
            ydiff = (y1 - y2, yy1 - yy2)

            div = aux.determinant(xdiff, ydiff)
            if div == 0:
                j += 1
                continue

            d = (aux.determinant((x1, y1), (x2, y2)),
                 aux.determinant((xx1, yy1), (xx2, yy2)))
            x = round(aux.determinant(d, xdiff) / div)
            y = round(aux.determinant(d, ydiff) / div)

            if x > img.wwidth or y > img.wheigth or x < 0 or y < 0:
                j += 1
                continue
            else:
                j += 1
                if aux.radius(last[0], last[1], x, y) > 10:
                    inter.append((x, y))
                    last = (x, y)
        i += 1

    canvas = np.zeros(img.warp3ch.shape, dtype='uint8')
    for i, p in enumerate(inter):
        cv2.circle(canvas, p, radius=5, color=(i*2, 0, 255-i*2), thickness=-1)
    cv2.addWeighted(img.warp3ch, 0.6, canvas, 0.4, 0, canvas)
    # aux.save(img, "interboard", canvas)

    inter = np.int32(np.round(inter))
    if len(inter) != 81:
        print("There should be exacly 81 intersections")
        exit(1)
    return inter


def mean_dist(distv, disth):
    medv1 = np.median(distv[:, 0])
    medv2 = np.median(distv[:, 1])
    print("medv1:", medv1)
    print("medv2:", medv2)
    medv = round((medv1 + medv2)/2)

    medh1 = np.median(disth[:, 0])
    medh2 = np.median(disth[:, 1])
    print("medh1:", medh1)
    print("medh2:", medh2)
    medh = round((medh1 + medh2)/2)

    return medv, medh


def wrong_lines(dist, med):
    rem = np.zeros(dist.shape[0], dtype='uint8')

    for i, d in enumerate(dist):
        if abs(d[0] - med) > 8:
            if abs(d[1] - med) > 8:
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
    print("adjusting vertical and horizontal lines...")

    print("calculating median distances...")
    distv, disth = get_distances(vert, hori)
    print("distv:", distv)
    print("disth:", disth)
    medv, medh = mean_dist(distv, disth)
    print("medv:", medv)
    print("medh:", medh)

    print("removing for sure wrong lines...")
    remv = wrong_lines(distv, medv)
    remh = wrong_lines(disth, medh)
    vert = vert[remv == 0]
    hori = hori[remh == 0]
    # aux.save_lines(img, "remwrong", vert, hori)

    print("updating median distances...")
    distv, disth = get_distances(vert, hori)
    medv, medh = mean_dist(distv, disth)

    print("chosing best lines...")
    cerv = right_lines(distv, medv)
    cerh = right_lines(disth, medh)
    vert = vert[cerv == 1]
    hori = hori[cerh == 1]
    # aux.save_lines(img, "right_lines", vert, hori)

    vert, hori = add_outer(vert, hori, medv, medh)
    # aux.save_lines(img, "add_outer", vert, hori)
    vert, hori = add_middle(vert, hori, medv, medh)
    # aux.save_lines(img, "add_middle", vert, hori)
    vert, hori = remove_extras(vert, hori)
    # aux.save_lines(img, "rem_extras", vert, hori)
    vert, hori = add_last_outer(vert, hori, medv, medh)
    # aux.save_lines(img, "last_outer", vert, hori)

    # aux.save_lines(img, "verthori1", vert, hori)
    if len(vert) != 9 or len(hori) != 9:
        print("There should be exactly 9 vertical and 9 horizontal lines")
        exit(1)
    return vert, hori


def calc_squares(img, inter):
    print("calculating squares corners...")
    inter = inter[inter[:, 0].argsort()]
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

    intersq[0, :] = interA[interA[:, 1].argsort()[::-1]]  # A
    intersq[1, :] = interB[interB[:, 1].argsort()[::-1]]  # B
    intersq[2, :] = interC[interC[:, 1].argsort()[::-1]]  # C
    intersq[3, :] = interD[interD[:, 1].argsort()[::-1]]  # D
    intersq[4, :] = interE[interE[:, 1].argsort()[::-1]]  # E
    intersq[5, :] = interF[interF[:, 1].argsort()[::-1]]  # F
    intersq[6, :] = interG[interG[:, 1].argsort()[::-1]]  # G
    intersq[7, :] = interH[interH[:, 1].argsort()[::-1]]  # H
    intersq[8, :] = interZ[interZ[:, 1].argsort()[::-1]]  # right

    squares = np.zeros((8, 8, 4, 2), dtype='int32')
    for i in range(0, 8):
        for j in range(0, 8):
            squares[i, j, 0] = intersq[i, j]
            squares[i, j, 1] = intersq[i+1, j]
            squares[i, j, 2] = intersq[i+1, j+1]
            squares[i, j, 3] = intersq[i, j+1]

    canvas = np.zeros(img.warp3ch.shape, dtype='uint8')
    cv2.drawContours(canvas, [squares[0, 0]], -1,  # A1
                     color=(255, 0, 0), thickness=img.thick)
    cv2.drawContours(canvas, [squares[4, 3]], -1,  # E4
                     color=(0, 255, 0), thickness=img.thick)
    cv2.drawContours(canvas, [squares[2, 4]], -1,  # C5
                     color=(0, 0, 255), thickness=img.thick)
    cv2.addWeighted(img.warp3ch, 0.6, canvas, 0.4, 0, canvas)
    # aux.save(img, "A1E4C5", canvas)

    return squares


def add_outer(vert, hori, medv, medh):
    print("adding missing outer lines...")
    while abs(vert[0, 0] - 0) > (medv + 5):
        x1 = vert[0, 0]-medv
        y1 = vert[0, 1]
        x2 = vert[0, 2]-medv
        y2 = vert[0, 3]
        new = np.array([[x1, y1, x2, y2, 0, 0]], dtype='int32')
        vert = np.append(vert, new, axis=0)
        vert = vert[vert[:, 0].argsort()]
    while abs(vert[-1, 0] - WARPED_LEN) > (medv + 5):
        x1 = vert[-1, 0] + medv
        y1 = vert[-1, 1]
        x2 = vert[-1, 2] + medv
        y2 = vert[-1, 3]
        new = np.array([[x1, y1, x2, y2, 0, 0]], dtype='int32')
        vert = np.append(vert, new, axis=0)
        vert = vert[vert[:, 0].argsort()]
    while abs(hori[0, 1] - 0) > (medh + 5):
        x1 = hori[0, 0]
        y1 = hori[0, 1] - medh
        x2 = hori[0, 2]
        y2 = hori[0, 3] - medh
        new = np.array([[x1, y1, x2, y2, 0, 0]], dtype='int32')
        hori = np.append(hori, new, axis=0)
        hori = hori[hori[:, 1].argsort()]
    while abs(hori[-1, 1] - WARPED_LEN) > (medh + 5):
        x1 = hori[-1, 0]
        y1 = hori[-1, 1] + medh
        x2 = hori[-1, 2]
        y2 = hori[-1, 3] + medh
        new = np.array([[x1, y1, x2, y2, 0, 0]], dtype='int32')
        hori = np.append(hori, new, axis=0)
        hori = hori[hori[:, 1].argsort()]

    return vert, hori


def add_middle(vert, hori, medv, medh):
    print("adding missing middle lines...")

    def _add_middle(lines, med, kind):
        i = 0
        while i < (len(vert) - 1):
            if abs(lines[i, kind] - lines[i+1, kind]) > (med*1.5):
                x1 = lines[i, 0]+med
                y1 = lines[i, 1]
                x2 = lines[i, 2]+med
                y2 = lines[i, 3]
                new = np.array([[x1, y1, x2, y2, 0, 0]], dtype='int32')
                lines = np.append(lines, new, axis=0)
                lines = lines[lines[:, kind].argsort()]
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
            d2 = abs(lines[-1, kind] - WARPED_LEN)
            if d1 < d2:
                lines = lines[1:]
            else:
                lines = lines[0:-1]
        elif num == 11:
            lines = lines[1:-1]
        elif num >= 12:
            print("There are 12 or more vertical lines")
            exit(1)
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
            d2 = abs(lines[-1, 0] - WARPED_LEN)
            if d1 > d2:
                if d1 >= med:
                    x1 = lines[0, 0]-med
                    y1 = lines[0, 1]
                    x2 = lines[0, 2]-med
                    y2 = lines[0, 3]
                    new = np.array([[x1, y1, x2, y2, 0, 0]], dtype='int32')
                else:
                    print("not enough space for inserting missing 9th line")
                    exit(1)
            else:
                if d2 >= med:
                    x1 = lines[-1, 0]+med
                    y1 = lines[-1, 1]
                    x2 = lines[-1, 2]+med
                    y2 = lines[-1, 3]
                    new = np.array([[x1, y1, x2, y2, 0, 0]], dtype='int32')
                else:
                    print("not enough space for inserting missing 9th line")
                    exit(1)

            lines = np.append(lines, new, axis=0)
            lines = lines[lines[:, kind].argsort()]
        return lines

    vert = _add_last_outer(vert, v, medv, 0)
    hori = _add_last_outer(hori, h, medh, 1)
    return vert, hori
