import numpy as np

import auxiliar as aux
import drawings as draw

WLEN = 640
DX = 50


def magic_dir(img, vert, hori):
    def _check_save(title):
        nonlocal lv, lh, vert, hori
        if lv != len(vert) or lh != len(hori):
            canvas = draw.lines(img.warp3ch, vert, hori)
            aux.save(img, title, canvas)
            lv, lh = len(vert), len(hori)
        return

    lv, lh = len(vert), len(hori)
    distv, disth = get_distances(vert, hori)
    medv, medh = aux.mean_dist(distv, disth)

    print("removing for sure wrong vertical lines...")
    vert = aux.wrong_lines(vert, distv, medv, tol=2)
    lv = len(vert)
    print("removing for sure wrong horizontal lines...")
    hori = aux.wrong_lines(hori, disth, medh, tol=2)
    lh = len(hori)
    _check_save("rem_wrong")
    return vert, hori, medv, medh


def rem_1011(img, vert, hori, medv, medh):
    tol = 2
    ww = img.bwidth
    hh = img.bheigth
    lv, lh = len(vert), len(hori)
    if lv == 9:
        vtol = medv + tol + DX
        if abs(vert[0, 0] - 0) > vtol and abs(vert[0, 2] - 0) > vtol:
            vert = vert[0:-1]
        elif abs(vert[-1, 0] - ww) > vtol and abs(vert[-1, 2] - ww) > vtol:
            vert = vert[1:]
    if lh == 9:
        htol = medh + tol + DX
        if abs(hori[0, 1] - 0) > htol and abs(hori[0, 3] - 0) > htol:
            hori = hori[0:-1]
        elif abs(hori[-1, 1] - hh) > htol and abs(hori[-1, 3] - hh) > htol:
            hori = hori[1:]

    canvas = draw.lines(img.gray3ch, vert, hori)
    aux.save(img, "after==9", canvas)
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


def right_lines(dist, med):
    tol = 8
    cer = np.zeros(dist.shape[0], dtype='uint8')

    for i, d in enumerate(dist):
        if abs(d[0] - med) < tol and abs(d[1] - med) < tol:
            cer[i] = 1
        else:
            cer[i] = 0
    return cer


def add_wouter(vert, hori, medv, medh):
    tol = 2
    print("adding missing outer lines...")
    while abs(vert[0, 0] - 0) > (medv + tol):
        x1 = vert[0, 0] - medv
        y1 = vert[0, 1]
        x2 = vert[0, 2] - medv
        y2 = vert[0, 3]
        new = np.array([[x1, y1, x2, y2, 0, 0]], dtype='int32')
        vert = np.append(vert, new, axis=0)
        vert = vert[np.argsort(vert[:, 0])]
    while abs(vert[-1, 0] - WLEN) > (medv + tol):
        x1 = vert[-1, 0] + medv
        y1 = vert[-1, 1]
        x2 = vert[-1, 2] + medv
        y2 = vert[-1, 3]
        new = np.array([[x1, y1, x2, y2, 0, 0]], dtype='int32')
        vert = np.append(vert, new, axis=0)
        vert = vert[np.argsort(vert[:, 0])]
    while abs(hori[0, 1] - 0) > (medh + tol):
        x1 = hori[0, 0]
        y1 = hori[0, 1] - medh
        x2 = hori[0, 2]
        y2 = hori[0, 3] - medh
        new = np.array([[x1, y1, x2, y2, 0, 0]], dtype='int32')
        hori = np.append(hori, new, axis=0)
        hori = hori[np.argsort(hori[:, 1])]
    while abs(hori[-1, 1] - WLEN) > (medh + tol):
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
        for i in range(1, len(lines) - 1):
            dprev = abs(lines[i, kind] - lines[i-1, kind])
            dnext = abs(lines[i, kind] - lines[i+1, kind])
            if dnext > (dprev * 1.5):
                dnext = round(dnext / 2)
                if kind == 0:
                    x1 = lines[i, 0] + dnext
                    y1 = lines[i, 1]
                    x2 = lines[i, 2] + dnext
                    y2 = lines[i, 3]
                else:
                    x1 = lines[i, 0]
                    y1 = lines[i, 1] + dnext
                    x2 = lines[i, 2]
                    y2 = lines[i, 3] + dnext
                new = np.array([[x1, y1, x2, y2, 0, 0]], dtype='int32')
                lines = np.append(lines, new, axis=0)
                lines = lines[np.argsort(lines[:, kind])]
        return lines

    vert = _add_middle(vert, medv, 0)
    hori = _add_middle(hori, medh, 1)
    return vert, hori


def remove_extras(vert, hori):
    print("removing extra outer lines...")
    if len(vert) <= 9 and len(hori) <= 9:
        return vert, hori

    def _rem_extras(lines, kind):
        ll = len(lines)
        if ll == 10:
            d1 = abs(lines[0, kind] - 0)
            d2 = abs(lines[-1, kind] - WLEN)
            if d1 < d2:
                lines = lines[1:]
            else:
                lines = lines[:-1]
        elif ll == 11:
            lines = lines[1:-1]
        elif ll >= 12:
            print("There are 12 or more lines")
            lines = lines[1:-1]
            lines = _rem_extras(lines, kind)
        return lines

    vert = _rem_extras(vert, 0)
    hori = _rem_extras(hori, 1)
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
