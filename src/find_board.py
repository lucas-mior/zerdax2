import cv2
import numpy as np
import math

import auxiliar as aux

from bundle_lines import bundle_lines

WARPED_LEN = 640


def find_board(img):
    print("finding select best lines...")
    print(f"img: {img}...")
    img = select_prepare(img)
    print(f"img: {img}...")
    img = select_lines(img)
    lines, img.broadcorners = magic_lines(img)
    inter = calc_intersections(img, lines[:, 0, :])
    img.corners = calc_corners(img, inter)
    img = perspective_transform(img)

    return img


def create_cannys(img, w=5, c_thrhg=220, c_thrhv=220, saveny=False):
    print("finding edges for gray, S, V images...")
    cannyG, img.cg0 = aux.find_canny(img, img.claheG, wmin=w, c_thrh=c_thrhg)
    cannyV, img.cv0 = aux.find_canny(img, img.claheV, wmin=w, c_thrh=c_thrhv)
    img.cg0 += 5
    img.cv0 += 5
    aux.save(img, "cannyG", cannyG)
    aux.save(img, "cannyV", cannyV)
    img.canny = cv2.bitwise_or(cannyG, cannyV)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img.canny = cv2.morphologyEx(img.canny, cv2.MORPH_CLOSE, k_close)
    return img


def select_lines(img):

    got_hough = False
    h_maxg = 50
    h_minl = h_minl0 = round((img.bwidth + img.bheigth)*0.21) - 40
    h_thrv = round(h_minl0 / 1.8)
    h_angl = np.pi / 360

    while h_angl <= (np.pi / 180) or h_minl >= (h_minl0 / 1.3):
        th = math.degrees(h_angl)
        lines = cv2.HoughLinesP(img.canny, 1,
                                h_angl, h_thrv, None, h_minl, h_maxg)
        if lines is None:
            h_minl = max(h_minl0 / 1.3, h_minl - 8)
            h_thrv = round(h_minl / 1.8)
            h_angl = min(np.pi/180, h_angl + np.pi/1800)
            continue
        if len(lines) >= 22:
            print(f"{len(lines)} lines @ {th:1=.4f}º, {h_thrv}, {h_minl}, {h_maxg}")
            lines = aux.radius_theta(lines)
            lines = filter_lines(img, lines)
            angles = lines_kmeans(img, lines)
            print("lines angles means:\n", angles, sep='')
            got_hough = True
            break

        print(f"{len(lines)} lines @ {th:1=.4f}º, {h_thrv}, {h_minl}, {h_maxg}")
        if h_angl >= (np.pi/180) and h_minl <= (h_minl0/1.3):
            break
        h_minl = max(h_minl0 / 1.3, h_minl - 4)
        h_thrv = round(h_minl / 1.8)
        h_angl = min(np.pi/180, h_angl + np.pi/3600)

    if not got_hough:
        print("select_lines failed")
        exit(1)

    canvas2 = np.zeros(img.gray3ch.shape, dtype='uint8')
    for line in lines:
        for x1, y1, x2, y2, r, t in line:
            canvas2 = cv2.line(canvas2, (x1, y1), (x2, y2),
                               color=(0, 255, 255), thickness=3)
    img.select = canvas2[:, :, 2]
    canvas2 = cv2.addWeighted(img.gray3ch, 0.5, canvas2, 0.5, 1)
    aux.save(img, "select", canvas2)

    img.select_lines = lines
    img.angles = angles
    img.slen = round(img.select_lines[:, 0, 4].min())
    return img


def calc_intersections(img, lines):
    print("calculating intersections...")
    inter = []

    i = 0
    for x1, y1, x2, y2, r, t in lines:
        l1 = [(x1, y1), (x2, y2)]
        j = 0
        for xx1, yy1, xx2, yy2, rr, tt in lines:
            l2 = [(xx1, yy1), (xx2, yy2)]
            if (x1, y1) == (xx1, yy1) and (x2, y2) == (xx2, yy2):
                continue

            if abs(t - tt) < 20 or abs(t - tt) > 160:
                continue

            xdiff = (l1[0][0] - l1[1][0], l2[0][0] - l2[1][0])
            ydiff = (l1[0][1] - l1[1][1], l2[0][1] - l2[1][1])

            div = aux.determinant(xdiff, ydiff)
            if div == 0:
                j += 1
                continue

            d = (aux.determinant(*l1), aux.determinant(*l2))
            x = round(aux.determinant(d, xdiff) / div)
            y = round(aux.determinant(d, ydiff) / div)

            if x > img.bwidth or y > img.bheigth or x < 0 or y < 0:
                j += 1
                continue
            else:
                j += 1
                inter.append((x, y))
        i += 1

    inter = np.int32(np.round(inter))
    canvas4 = np.zeros(img.gray3ch.shape, dtype='uint8')
    for p in inter:
        canvas4 = cv2.circle(canvas4, p, radius=7,
                             color=(0, 0, 255), thickness=-1)
    canvas4 = cv2.addWeighted(img.gray3ch, 0.5, canvas4, 0.5, 1)
    aux.save(img, "intersections", canvas4)

    return inter


def magic_lines(img):
    print("finding all lines of board...")
    img = magic_prepare(img)

    broadcorners = False
    got_hough = False
    force = 1.2
    h_maxg = 100
    h_minl = h_minl0 = img.slen
    h_thrv = round(h_minl / force)
    h_angl = np.pi / 480
    h_a = math.degrees(h_angl)

    def _update_magic(force):
        nonlocal h_minl, h_thrv
        print(f"force: {force:1=.3f}")
        h_minl = h_minl0
        h_thrv = round(h_minl / force)
        return

    incr = 32
    while h_minl >= (img.slen/1.5):
        l1 = l2 = ll = 0
        lines = cv2.HoughLinesP(img.test, 1,
                                h_angl, h_thrv, None, h_minl, h_maxg)
        if lines is None:
            h_minl = max(img.slen/1.4, h_minl - incr)
            h_thrv = round(h_minl / force)
            continue

        if len(lines) < 18:
            h_minl = max(img.slen/1.4, h_minl - incr)
            h_thrv = round(h_minl / force)
            continue

        lines = aux.radius_theta(lines)
        lines = filter_lines(img, lines)
        lines = filter_angles(img, lines)
        if len(lines) < 16:
            h_minl = max(img.slen/1.4, h_minl - incr/2)
            h_thrv = round(h_minl / force)
            continue

        lines = bundle_lines(lines)
        lines = aux.radius_theta(lines)
        ll = len(lines)
        if ll >= 18:
            dir1, dir2 = split_lines(img, lines)
            l1 = len(dir1)
            l2 = len(dir2)
            if 22 <= ll <= 24 and (11 <= l1 <= 13 and 11 <= l2 <= 13):
                print(f"{len(lines)} # [{l1}][{l2}] @ {h_a:1=.3f}º,{h_thrv},{h_minl},{h_maxg}")
                got_hough = True
                break
            if ll >= 25 and (l1 >= 14 or l2 >= 14):
                h_minl += 30
                h_thrv = round(h_minl / force)
                incr = 8
                continue

        print(f"{len(lines)} # [{l1}][{l2}] @ {h_a:1=.3f}º,{h_thrv},{h_minl},{h_maxg}")
        h_minl -= incr
        h_thrv = round(h_minl / force)
        if h_minl <= (img.slen/1.4):
            if force <= 1.5:
                force += 0.1
                _update_magic(force)
            elif force <= 1.8 and (l1 < 10 or l2 < 10):
                force += 0.1
                _update_magic(force)

    if l1 > 0 and l2 > 0:
        aux.save_lines(img, "hough_magic", dir1, dir2, warp=False)

    dummy = np.copy(img.select_lines[:, :, 0:6])
    lines = np.append(lines, dummy, axis=0)
    lines = filter_angles(img, lines)

    if not got_hough:
        if l1 < 10 or l2 < 10:
            print(f"magic_lines() failed @ {180*(h_angl/np.pi)}, {h_thrv}, {h_minl}, {h_maxg}")
            aux.save(img, "last_test", img.test)
            exit(1)
        else:
            broadcorners = True
            aux.save(img, "last_test", img.test)
            print("could not find 11 lines in at least one side."
                  "Trying with 10 on both sides.")

    return lines, broadcorners


def filter_lines(img, lines):
    rem = np.zeros(lines.shape[0], dtype='uint8')

    i = 0
    for line in lines:
        for x1, y1, x2, y2, r, t in line:
            if x1 < 28 and x2 < 28 or y1 < 28 and y2 < 28:
                rem[i] = 1
            elif (img.bwidth - x1) < 28 and (img.bwidth - x2) < 28 or (img.bheigth - y1) < 28 and (img.bheigth - y2) < 28:
                rem[i] = 1
            elif (x1 < 28 or (img.bwidth - x1) < 28) and (y2 < 28 or (img.bheigth - y2) < 28):
                rem[i] = 1
            elif (x2 < 28 or (img.bwidth - x2) < 28) and (y1 < 28 or (img.bheigth - y1) < 28):
                rem[i] = 1
            else:
                rem[i] = 0
        i += 1

    A = lines[rem == 0]
    lines = A
    return lines


def filter_angles(img, lines, tol=15):
    rem = np.zeros(lines.shape[0], dtype='uint8')

    i = 0
    for line in lines:
        for x1, y1, x2, y2, r, t in line:
            if abs(t - img.angles[0]) > tol and abs(t - img.angles[1]) > tol:
                if len(img.angles) == 2:
                    rem[i] = 1
                elif abs(t - img.angles[2]) > tol:
                    rem[i] = 1
                else:
                    rem[i] = 0
            else:
                rem[i] = 0
        i += 1

    A = lines[rem == 0]
    lines = A
    return lines


def lines_kmeans(img, lines):
    lines = np.float32(lines)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compact, labels, centers = cv2.kmeans(lines[:, :, 5], 3, None,
                                          criteria, 10, flags)

    d1 = abs(centers[0] - centers[1])
    d2 = abs(centers[0] - centers[2])
    d3 = abs(centers[1] - centers[2])

    dd1 = d1 < 22.5 and d2 > 22.5 and d3 > 22.5
    dd2 = d2 < 22.5 and d1 > 22.5 and d3 > 22.5
    dd3 = d3 < 22.5 and d1 > 22.5 and d2 > 22.5

    if dd1 or dd2 or dd3:
        compact, labels, centers = cv2.kmeans(lines[:, :, 5], 2, None,
                                              criteria, 10, flags)

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

    centers = np.int32(np.round(centers))
    return centers


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

    if img.broadcorners:
        BR, BL, TR, TL = broad_corners(img, BR, BL, TR, TL)

    canvas4 = np.copy(img.gray3ch) * 0
    canvas4 = cv2.circle(canvas4, BR, radius=9,
                         color=(255, 0, 0), thickness=-1)
    canvas4 = cv2.circle(canvas4, BL, radius=9,
                         color=(0, 255, 0), thickness=-1)
    canvas4 = cv2.circle(canvas4, TR, radius=9,
                         color=(0, 0, 255), thickness=-1)
    canvas4 = cv2.circle(canvas4, TL, radius=9,
                         color=(255, 255, 255), thickness=-1)

    canvas4 = cv2.addWeighted(img.gray3ch, 0.5, canvas4, 0.5, 1)
    aux.save(img, "corners2", canvas4)

    corners = np.array([BR, BL, TR, TL])
    print("board corners:\n", corners, sep='')

    return corners


def perspective_transform(img):
    print("transforming perspective...")
    BR = img.corners[0]
    BL = img.corners[1]
    TR = img.corners[2]
    TL = img.corners[3]
    orig_points = np.array(((TL[0], TL[1]), (TR[0], TR[1]),
                            (BR[0], BR[1]), (BL[0], BL[1])), dtype="float32")

    width = WARPED_LEN
    height = WARPED_LEN
    img.wwidth = width
    img.wheigth = width

    newshape = np.array([[0, 0], [width-1, 0],
                        [width-1, height-1], [0, height-1]], dtype="float32")
    print("creating transform matrix...")
    img.warpMatrix = cv2.getPerspectiveTransform(orig_points, newshape)
    _, img.warpInvMatrix = cv2.invert(img.warpMatrix)
    print("warping image...")
    img.wg = cv2.warpPerspective(img.gray, img.warpMatrix, (width, height))
    img.wv = cv2.warpPerspective(img.V, img.warpMatrix, (width, height))

    return img


def split_lines(img, lines):
    lines = np.float32(lines)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compact, labels, centers = cv2.kmeans(lines[:, :, 5], 3, None,
                                          criteria, 10, flags)

    A = lines[labels == 0]
    B = lines[labels == 1]

    d1 = abs(centers[0] - centers[1])
    d2 = abs(centers[0] - centers[2])
    d3 = abs(centers[1] - centers[2])

    dd1 = d1 < 22.5 and d2 > 22.5 and d3 > 22.5
    dd2 = d2 < 22.5 and d1 > 22.5 and d3 > 22.5
    dd3 = d3 < 22.5 and d1 > 22.5 and d2 > 22.5

    if dd1 or dd2 or dd3:
        compact, labels, centers = cv2.kmeans(lines[:, :, 5], 2, None,
                                              criteria, 10, flags)
        A = lines[labels == 0]
        B = lines[labels == 1]

    if len(centers) == 3:
        # Redo kmeans using absolute inclination
        lines = np.float32(aux.radius_theta(lines, absol=True))
        compact, labels, centers = cv2.kmeans(lines[:, :, 5], 2, None,
                                              criteria, 10, flags)
        A = lines[labels == 0]
        B = lines[labels == 1]

    return np.int32(A), np.int32(B)


def magic_prepare(img):
    print("preparing image for magic...")
    img = create_cannys(img, w=8.5, saveny=False)
    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img.canny = cv2.morphologyEx(img.canny, cv2.MORPH_DILATE, k_dil)
    aux.save(img, "canny9", img.canny)
    aux.save(img, "fedges", img.fedges)

    mid = round(img.bheigth/2)
    end = img.bheigth + 1
    up = img.canny[0:mid, :]
    down = img.canny[mid:end, :]
    downf = img.fedges[mid:end, :]
    down = cv2.bitwise_and(down, downf)
    down = cv2.morphologyEx(down, cv2.MORPH_DILATE, k_dil)
    img.test = np.concatenate((up, down), axis=0)
    aux.save(img, "andfedges", img.test)

    img.test = cv2.bitwise_or(img.test, img.select)
    k_clo = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img.test = cv2.morphologyEx(img.test, cv2.MORPH_CLOSE, k_clo)
    aux.save(img, "ny+select-closed", img.test)
    return img


def select_prepare(img):
    print("finding edges for selecting lines...")
    img = create_cannys(img, w=7, saveny=False)
    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img.canny = cv2.morphologyEx(img.canny, cv2.MORPH_DILATE, k_dil)
    img.canny = cv2.morphologyEx(img.canny, cv2.MORPH_CLOSE, k_dil)
    aux.save(img, "canny7_select", img.canny)

    return img


def broad_corners(img, BR, BL, TR, TL):
    print("adding margin for corners...")
    BR[0] = min(img.bwidth-1,  BR[0]+5)
    BR[1] = min(img.bheigth-1, BR[1]+5)
    BL[0] = max(0,             BL[0]-5)
    BL[1] = min(img.bheigth-1, BL[1]+5)
    TR[0] = min(img.bwidth-1,  TR[0]+5)
    TR[1] = max(0,             TR[1]-5)
    TL[0] = max(0,             TL[0]-5)
    TL[1] = max(0,             TL[1]-5)
    return BR, BL, TR, TL
