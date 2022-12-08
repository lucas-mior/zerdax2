import cv2
import numpy as np
import math

import auxiliar as aux

from bundle_lines import bundle_lines
import lffilter as lf


def find_board(img):
    img = pre_process(img)
    img = find_region(img)
    img = bound_region(img)
    img = reduce_box(img)
    img = black_space(img)

    img = select_lines(img)
    lines, img.broadcorners = magic_lines(img)
    inter = calc_intersections(img, lines[:, 0, :])
    img.corners = calc_corners(img, inter)
    img = perspective_transform(img)

    return img


def pre_process(img):
    print("pre processing image...")
    print("applying filter to image...")
    img.G = cv2.GaussianBlur(img.gray, (7, 7), 0.3)
    img.V = cv2.GaussianBlur(img.V, (7, 7), 0.3)
    # img.G = lf.ffilter(img.gray)
    # img.V = lf.ffilter(img.V)

    print("applying distributed histogram equalization to image...")
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
    img.claheG = clahe.apply(img.G)
    img.claheV = clahe.apply(img.V)

    print("applying filter again...")
    img.claheG = lf.ffilter(img.claheG)
    img.claheV = lf.ffilter(img.claheV)

    print("creating insider...")
    img.inside = np.ones(img.G.shape, dtype='uint8') * 255

    return img


def create_cannys(img, w=5, c_thrhg=220, c_thrhv=220, saveny=False):
    aux.logprint(img, "finding edges for gray, S, V images...")
    cannyG, img.cg0 = aux.find_canny(img, img.claheG, wmin=w, c_thrh=c_thrhg)
    cannyV, img.cv0 = aux.find_canny(img, img.claheV, wmin=w, c_thrh=c_thrhv)
    img.cg0 += 5
    img.cv0 += 5
    if saveny:
        aux.save(img, "cannyG", cannyG)
        aux.save(img, "cannyV", cannyV)
    img.canny = cv2.bitwise_or(cannyG, cannyV)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img.canny = cv2.morphologyEx(img.canny, cv2.MORPH_CLOSE, k_close)
    img.canny = cv2.bitwise_and(img.canny, img.inside)
    return img


def bound_region(img):
    print("cropping image to fit only board region...")
    x, y, w, h = cv2.boundingRect(img.hullxy)
    margin = 10
    print(f"adding {margin} pixels margin...")
    x0 = max(y-margin, 0)
    x1 = min(y+h+margin, img.width)+1
    y0 = max(x-margin, 0)
    y1 = max(x+w+margin, img.heigth)+1

    def _crop(image):
        return image[x0:x1, y0:y1]

    img.G = _crop(img.G)
    img.V = _crop(img.V)
    img.claheG = _crop(img.claheG)
    img.claheV = _crop(img.claheV)
    img.fedges = _crop(img.fedges)
    img.dcont = _crop(img.dcont)
    img.gray = _crop(img.gray)
    img.BGR = _crop(img.BGR)
    img.gray3ch = _crop(img.gray3ch)
    return img


def reduce_box(img):
    print("reducing images to default size...")
    img.hwidth = 900
    img.hfact = img.hwidth / img.gray.shape[1]
    img.hheigth = round(img.hfact * img.gray.shape[0])
    img.harea = img.hwidth * img.hheigth
    nsh = (img.hwidth, img.hheigth)
    innsh = (img.hwidth - 10, img.hheigth - 10)

    print(f"reducing all images to {img.hwidth} width")
    img.G = cv2.resize(img.G, nsh)
    img.V = cv2.resize(img.V, nsh)
    img.claheG = cv2.resize(img.claheG, nsh)
    img.claheV = cv2.resize(img.claheV, nsh)
    img.dcont = cv2.resize(img.dcont, nsh)
    img.fedges = cv2.resize(img.fedges, nsh)
    img.gray = cv2.resize(img.gray, nsh)
    img.BGR = cv2.resize(img.BGR, nsh)
    img.gray3ch = cv2.resize(img.gray3ch, nsh)
    img.inside = cv2.resize(img.inside, innsh)

    return img


def black_space(img):
    def _mk_border(image, dx=20):
        return cv2.copyMakeBorder(image, dx, dx, dx, dx,
                                  cv2.BORDER_CONSTANT, None, value=0)
    print("adding black space around images...")
    img.hwidth += 40
    img.hheigth += 40
    img.G = _mk_border(img.G)
    img.V = _mk_border(img.V)
    img.claheG = _mk_border(img.claheG)
    img.claheV = _mk_border(img.claheV)
    img.dcont = _mk_border(img.dcont)
    img.fedges = _mk_border(img.fedges)
    img.gray = _mk_border(img.gray)
    img.gray3ch = _mk_border(img.gray3ch)
    img.BGR = _mk_border(img.BGR)
    img.inside = _mk_border(img.inside, dx=25)

    return img


def find_region(img):
    print("finding region containing chess board...")
    got_hull = False
    h = False
    Wc = 5
    W0 = 12
    Amin = round(0.5 * img.area)
    A0 = round(0.1 * img.area)
    img.help = np.copy(img.G) * 0
    img.cg0 = 200
    img.cv0 = 200
    while Wc <= W0 or Amin >= A0:
        aux.logprint(img, f"Área mínima: {Amin}")
        aux.logprint(img, f"Canny Wc: {Wc}")
        if Wc >= 9:
            h = True
        img = create_cannys(img, w=Wc, c_thrhg=img.cg0, c_thrhv=img.cv0)
        img, a = find_morph(img, h)

        canvas5 = np.zeros(img.gray3ch.shape, dtype='uint8')
        canvas5 = cv2.drawContours(canvas5, img.cont, -1,
                                   color=(255, 0, 0), thickness=1)
        canvas5 = cv2.drawContours(canvas5, [img.hullxy], -1,
                                   color=(0, 255, 0), thickness=1)
        img.help = cv2.bitwise_or(canvas5[:, :, 0], canvas5[:, :, 1])

        if a > Amin:
            aux.logprint(img, f"{a} > {Amin} : {a/Amin}")
            got_hull = True
            break
        else:
            aux.logprint(img, f"{a} < {Amin} : {a/Amin}")

        Amin = max(A0, round(Amin - 0.02*img.area))
        Wc = min(W0, Wc + 0.5)

    img.dcont = canvas5[:, :, 0]
    canvas5 = cv2.addWeighted(img.gray3ch, 0.5, canvas5, 0.5, 1)
    aux.save(img, "edges", img.edges)
    aux.save(img, "contours", canvas5)

    if not got_hull:
        print("finding board region failed")
        exit(1)

    return img


def find_morph(img, h=False):
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kd = 10
    kx = kd+round(kd/3)
    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (kd, kx))
    img.dilate = cv2.morphologyEx(img.claheG, cv2.MORPH_DILATE, k_dil)
    img.divide = cv2.divide(img.claheG, img.dilate, scale=255)
    edges = cv2.threshold(img.divide, 0, 255, cv2.THRESH_OTSU)[1]
    edges = cv2.bitwise_not(edges)
    edges = cv2.morphologyEx(edges, cv2.MORPH_ERODE, ko, iterations=1)
    img.fedges = np.copy(edges)
    edges = cv2.bitwise_or(edges, img.canny)
    if h:
        edges = cv2.bitwise_or(edges, img.help)
    contours, _ = cv2.findContours(edges,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in contours]
    img.cont = contours[np.argmax(areas)]
    img.hullxy = cv2.convexHull(img.cont)
    a = cv2.contourArea(img.hullxy)
    img.edges = edges

    return img, a


def select_lines(img):
    print("finding select best lines...")
    img = select_prepare(img)

    got_hough = False
    h_maxg = 50
    h_minl = h_minl0 = round((img.hwidth + img.hheigth)*0.21) - 40
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
            aux.logprint(img, f"{len(lines)} lines @ {th:1=.4f}º, {h_thrv}, {h_minl}, {h_maxg}")
            lines = aux.radius_theta(lines)
            lines = filter_lines(img, lines)
            angles = lines_kmeans(img, lines)
            print("lines angles means:\n", angles, sep='')
            got_hough = True
            break

        aux.logprint(img, f"{len(lines)} lines @ {th:1=.4f}º, {h_thrv}, {h_minl}, {h_maxg}")
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

            if x > img.hwidth or y > img.hheigth or x < 0 or y < 0:
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
                aux.logprint(img, f"{len(lines)} # [{l1}][{l2}] @ {h_a:1=.3f}º,{h_thrv},{h_minl},{h_maxg}")
                got_hough = True
                break
            if ll >= 25 and (l1 >= 14 or l2 >= 14):
                h_minl += 30
                h_thrv = round(h_minl / force)
                incr = 8
                continue

        aux.logprint(img, f"{len(lines)} # [{l1}][{l2}] @ {h_a:1=.3f}º,{h_thrv},{h_minl},{h_maxg}")
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
    rem = np.empty(lines.shape[0], dtype='uint8')

    i = 0
    for line in lines:
        for x1, y1, x2, y2, r, t in line:
            if x1 < 28 and x2 < 28 or y1 < 28 and y2 < 28:
                rem[i] = 1
            elif (img.hwidth - x1) < 28 and (img.hwidth - x2) < 28 or (img.hheigth - y1) < 28 and (img.hheigth - y2) < 28:
                rem[i] = 1
            elif (x1 < 28 or (img.hwidth - x1) < 28) and (y2 < 28 or (img.hheigth - y2) < 28):
                rem[i] = 1
            elif (x2 < 28 or (img.hwidth - x2) < 28) and (y1 < 28 or (img.hheigth - y1) < 28):
                rem[i] = 1
            else:
                rem[i] = 0
        i += 1

    A = lines[rem == 0]
    lines = A
    return lines


def filter_angles(img, lines, tol=15):
    rem = np.empty(lines.shape[0], dtype='uint8')

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
    psum = np.empty((inter.shape[0], 3), dtype='int32')
    psub = np.empty((inter.shape[0], 3), dtype='int32')

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

    width = 512
    height = 512
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

    mid = round(img.hheigth/2)
    end = img.hheigth + 1
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
    BR[0] = min(img.hwidth-1,  BR[0]+5)
    BR[1] = min(img.hheigth-1, BR[1]+5)
    BL[0] = max(0,             BL[0]-5)
    BL[1] = min(img.hheigth-1, BL[1]+5)
    TR[0] = min(img.hwidth-1,  TR[0]+5)
    TR[1] = max(0,             TR[1]-5)
    TL[0] = max(0,             TL[0]-5)
    TL[1] = max(0,             TL[1]-5)
    return BR, BL, TR, TL
