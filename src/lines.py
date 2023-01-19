import numpy as np
from numpy.linalg import det
import cv2
import logging as log

import auxiliar as aux
import constants as consts
import drawings as draw
from lines_bundle import lines_bundle

minlen0 = consts.min_line_length
bonus = 0


def find_lines(img, canny):
    log.info("finding all lines of board...")
    global bonus
    canny3ch = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    min_before_split = consts.min_lines_before_split

    angle = consts.hough_angle_resolution
    tangle = np.deg2rad(angle)  # radians
    minlen = minlen0
    maxgap = round(minlen0 / consts.hough_maxgap_factor)
    tvotes = round(minlen0)
    ll = lv = lh = 0
    while (lv < 9 or lh < 9) and tvotes > (minlen0 / 1.4):
        minlen = max(minlen - 8, minlen0 / 1.2)
        tvotes -= 12
        lines = cv2.HoughLinesP(canny, 1,
                                tangle, tvotes, None, minlen, maxgap)
        if (ll := lines.shape[0]) < min_before_split:
            log.debug(f"{ll} @ {angle}, {tvotes}, {minlen}, {maxgap}")
            continue
        lines_hough = lines[:, 0, :]
        lines = lines_bundle(lines_hough)
        if aux.debugging():
            canvas = draw.lines(canny3ch, lines_hough)
            aux.save("hough_lines", canvas)
            canvas = draw.lines(canny3ch, lines)
            aux.save("lines_bundled", canvas)
        lines, _ = length_theta(lines)
        vert, hori = split_lines(lines)
        lv, lh = check_save(canny3ch, "split_lines", vert, hori, 0, 0)
        vert, hori = filter_byangle(vert, hori)
        lv, lh = check_save(canny3ch, "filter_byangle", vert, hori, lv, lh)
        vert, hori = sort_lines(vert, hori)
        lv, lh = check_save(canny3ch, "sort_lines", vert, hori, 0, 0)
        ll = lv + lh
        log.info(f"{ll} # [{lv}][{lh}] @",
                 f"{angle}º, {tvotes}, {minlen}, {maxgap}")

    if lv < 9 or lh < 9:
        log.warning("Less than 9 lines find in at least one direction")
        canvas = draw.lines(canny3ch, vert, hori)
        aux.save(f"canny{lv=}_{lh=}", canvas)

    vert, hori = shorten_byinter(img.bwidth, img.bheigth, vert, hori)
    lv, lh = check_save(canny3ch, "shorten_byinter", vert, hori, -1, -1)
    vert, hori = add_outer(img.bwidth, img.bheigth, vert, hori)
    lv, lh = check_save(canny3ch, "add_outer", vert, hori, lv, lh)
    vert, hori = sort_lines(vert, hori)
    lv, lh = check_save(canny3ch, "sort_lines", vert, hori, -1, -1)
    vert, hori = shorten_byinter(img.bwidth, img.bheigth, vert, hori)
    lv, lh = check_save(canny3ch, "shorten_byinter", vert, hori, -1, -1)
    vert, hori = remove_extras(vert, hori, img.bwidth, img.bheigth)
    lv, lh = check_save(canny3ch, "remove_extras", vert, hori, lv, lh)
    vert, hori = add_middle(vert, hori)
    lv, lh = check_save(canny3ch, "add_middle", vert, hori, lv, lh)
    vert, hori = sort_lines(vert, hori)
    lv, lh = check_save(canny3ch, "sort_lines", vert, hori, -1, -1)
    vert, hori = remove_extras(vert, hori, img.bwidth, img.bheigth)
    lv, lh = check_save(canny3ch, "remove_extras", vert, hori, lv, lh)

    if aux.debugging():
        canvas = draw.lines(img.gray3ch, vert, hori)
        aux.save("find_lines", canvas)
    return vert, hori


def add_outer(ww, hh, vert, hori):
    log.info("adding missing outer lines...")
    tol = consts.outer_tolerance

    def _add_outer(lines, k):
        for runs in range(2):
            lines = calc_outer(lines, tol, 0, k, ww, hh)
            lines = calc_outer(lines, tol, -1, k, ww, hh)
        return lines

    vert = _add_outer(vert, 0)
    hori = _add_outer(hori, 1)
    return vert, hori


def add_middle(vert, hori):
    log.info("adding missing middle lines...")
    tol = consts.middle_tolerance

    def _append(lines, i, x, y, kind):
        x0, x1 = x
        y0, y1 = y
        line = (x0, y0, x1, y1)
        new = np.array([[x0, y0, x1, y1,
                         length(line), theta(line), 0]], dtype='int32')
        lines = np.append(lines, new, axis=0)
        lines, _ = sort_lines(lines, k=kind)
        return lines

    def _add_middle(lines, kind):
        dthis = (abs(lines[0, kind] - lines[1, kind])
                 + abs(lines[0, kind+2] - lines[1, kind+2])) / 2
        dnext0 = (abs(lines[1, kind] - lines[2, kind])
                  + abs(lines[1, kind+2] - lines[2, kind+2])) / 2
        dnext1 = (abs(lines[2, kind] - lines[3, kind])
                  + abs(lines[2, kind+2] - lines[3, kind+2])) / 2
        x = (lines[1, 0] - lines[2, 0] + lines[1, 0],
             lines[1, 2] - lines[2, 2] + lines[1, 2])
        y = (lines[1, 1] - lines[2, 1] + lines[1, 1],
             lines[1, 3] - lines[2, 3] + lines[1, 3])
        if dthis > (dnext0*tol) and dthis > (dnext1*tol):
            lines = _append(lines, 0, x, y, kind)
            i = 0
        for i in range(1, len(lines) - 2):
            dprev = (abs(lines[i+0, kind] - lines[i-1, kind])
                     + abs(lines[i+0, kind+2] - lines[i-1, kind+2])) / 2
            dthis = (abs(lines[i+0, kind] - lines[i+1, kind])
                     + abs(lines[i+0, kind+2] - lines[i+1, kind+2])) / 2
            dnext = (abs(lines[i+1, kind] - lines[i+2, kind])
                     + abs(lines[i+1, kind+2] - lines[i+2, kind+2])) / 2
            if dthis > (dprev*tol) and dthis > (dnext*tol):
                x = (round((lines[i, 0] + lines[i+1, 0])/2),
                     round((lines[i, 2] + lines[i+1, 2])/2))
                y = (round((lines[i, 1] + lines[i+1, 1])/2),
                     round((lines[i, 3] + lines[i+1, 3])/2))
                lines = _append(lines, i, x, y, kind)
        return lines

    def _adds(lines, kind):
        lines = _add_middle(lines, kind)
        lines = np.flip(lines, axis=0)
        lines = _add_middle(lines, kind)
        if len(lines) <= 9:
            lines = _add_middle(lines, kind)
            lines = np.flip(lines, axis=0)
            lines = _add_middle(lines, kind)
        return lines
    vert = _adds(vert, 0)
    hori = _adds(hori, 1)
    return vert, hori


def remove_extras(vert, hori, ww, hh):
    log.info("removing extra outer lines...")
    if (lv := vert.shape[0]) > 9:
        vert = rem_extras(vert, lv, k=0, dd=ww)
    if (lh := hori.shape[0]) > 9:
        hori = rem_extras(hori, lh, k=1, dd=hh)

    return vert, hori


def split_lines(lines):
    log.info("spliting lines into vertical and horizontal...")
    if (lines.shape[1] < 6):
        lines, _ = length_theta(lines)
    lines = np.array(lines, dtype='float32')

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compact, labels, centers = cv2.kmeans(lines[:, 5], 3, None,
                                          criteria, 15, flags)
    labels = np.ravel(labels)

    d0 = abs(centers[0] - centers[1])
    d1 = abs(centers[0] - centers[2])
    d2 = abs(centers[1] - centers[2])

    maxdiff = consts.angles_max_diff
    dd0 = d0 < maxdiff and d1 > maxdiff and d2 > maxdiff
    dd1 = d1 < maxdiff and d0 > maxdiff and d2 > maxdiff
    dd2 = d2 < maxdiff and d0 > maxdiff and d1 > maxdiff

    if dd0 or dd1 or dd2:
        compact, labels, centers = cv2.kmeans(lines[:, 5], 2, None,
                                              criteria, 15, flags)
        labels = np.ravel(labels)
        A = lines[labels == 0]
        B = lines[labels == 1]
    else:
        # redo kmeans using absolute inclination
        lines, _ = length_theta(lines, abs_angle=True)
        lines = np.array(lines, dtype='float32')
        compact, labels, centers = cv2.kmeans(lines[:, 5], 2, None,
                                              criteria, 15, flags)
        labels = np.ravel(labels)
        A = lines[labels == 0]
        B = lines[labels == 1]

    if abs(centers[1]) < abs(centers[0]):
        vert = np.array(A, dtype='int32')
        hori = np.array(B, dtype='int32')
    else:
        vert = np.array(B, dtype='int32')
        hori = np.array(A, dtype='int32')
    nvert = []
    for line in vert:
        x0, y0, x1, y1 = line[:4]
        if y0 > y1:
            a1, b1 = x0, y0
            x0, y0 = x1, y1
            x1, y1 = a1, b1
        line = [x0, y0, x1, y1, line[4], line[5]]
        nvert.append(line)
    vert = np.array(nvert, dtype='int32')
    return vert, hori


def sort_lines(vert, hori=None, k=0):
    log.debug("sorting lines by position and respective direction...")

    def _create(lines, kind):
        dummy = np.zeros((lines.shape[0], 7), dtype='int32')
        dummy[:, 0:6] = lines[:, 0:6]
        lines = dummy

        for i, line in enumerate(lines):
            inter = calc_intersection(line, kind=kind)
            lines[i, 6] = inter[kind]
        return lines

    if hori is not None:
        vert = _create(vert, kind=0)
        vert = vert[np.argsort(vert[:, 6])]
        hori = _create(hori, kind=1)
        hori = hori[np.argsort(hori[:, 6])]
    else:
        vert = _create(vert, kind=k)
        vert = vert[np.argsort(vert[:, 6])]
    return vert, hori


def filter_byangle(vert, hori=None, tol=15):
    log.info("filtering lines by angle accoring to direction...")
    tol = consts.angle_tolerance

    def _filter(lines):
        rem = np.zeros(lines.shape[0], dtype='uint8')
        angle = np.median(lines[:, 5])

        for i, line in enumerate(lines):
            if abs(line[5] - angle) > tol:
                rem[i] = 1
            else:
                rem[i] = 0
        return lines[rem == 0]

    vert = _filter(vert)
    if hori is not None:
        hori = _filter(hori)
    return vert, hori


def calc_outer(lines, tol, where, k, ww, hh):
    if k == 0:
        dd = ww
    else:
        dd = hh
    if where == 0:
        ee = 0
        other = 1
    elif where == -1:
        ee = dd
        other = -2

    line0 = lines[where]
    line1 = lines[other]

    dx = (line0[0] - line1[0], line0[2] - line1[2])
    dy = (line0[1] - line1[1], line0[3] - line1[3])
    if abs(line0[k] - ee) > tol and abs(line0[k+2] - ee) > tol:
        x0 = line0[0] + dx[0]
        y0 = line0[1] + dy[0]
        x1 = line0[2] + dx[1]
        y1 = line0[3] + dy[1]
        new = np.array([[x0, y0, x1, y1, 0, 0, 0]], dtype='int32')
        inters = limit_bydims(new[0][:4], ww, hh)
        if len(inters) != 2:
            return lines
        if inters[0, k] <= dd and inters[1, k] <= dd:
            x0, y0, x1, y1 = np.ravel(inters)
            line = (x0, y0, x1, y1)
            minlen = consts.min_line_length / 1.5
            if (r := length(line)) >= minlen:
                new = np.array([[x0, y0, x1, y1,
                                 r, theta(line), 0]], dtype='int32')
                lines = np.append(lines, new, axis=0)
                lines, _ = sort_lines(lines, k=k)
    return lines


def calc_inters(line, ww, hh):
    i0 = calc_intersection(line, ww, hh, kind=0)
    i1 = calc_intersection(line, ww, hh, kind=1)
    i2 = calc_intersection(line, ww, hh, kind=2)
    i3 = calc_intersection(line, ww, hh, kind=3)
    return np.array([i0, i1, i2, i3], dtype='int32')


def shorten_byinter(ww, hh, vert, hori=None):
    inters = calc_intersections(vert, hori)

    def _shorten(lines):
        nlines = []
        for i, inter in enumerate(inters):
            line = lines[i]
            a, b = inter[0], inter[-1]
            new = np.array([[a[0], a[1], b[0], b[1], 0, 0, 0]], dtype='int32')
            limit = limit_bydims(new[0, :4], ww, hh)
            limit = np.ravel(limit)
            if length(limit) < length(new[0, :4]):
                x0, y0, x1, y1 = limit[:4]
            else:
                x0, y0, x1, y1 = new[0, :4]
            new = x0, y0, x1, y1, line[4], line[5], line[6]
            nlines.append(new)
        lines = np.array(nlines, dtype='int32')
        return lines

    vert = _shorten(vert)
    if hori is not None:
        inters = np.transpose(inters, axes=(1, 0, 2))
        hori = _shorten(hori)
    return vert, hori


def limit_bydims(inters, ww, hh):
    inters = calc_inters(inters, ww, hh)
    inters = inters[(inters[:, 0] >= 0) & (inters[:, 1] >= 0) &
                    (inters[:, 0] <= ww) & (inters[:, 1] <= hh)]
    return inters


def rem_extras(lines, ll, k, dd):
    tol = consts.outer_tolerance
    if ll == 10:
        d10 = abs(lines[1, k] - 0)
        d11 = abs(lines[1, k+2] - 0)
        d1 = min(d10, d11)
        d20 = abs(lines[-2, k] - dd)
        d21 = abs(lines[-2, k+2] - dd)
        d2 = min(d20, d21)
        if d1 > tol and d2 < tol:
            lines = lines[:-1]
        elif d2 > tol and d1 < tol:
            lines = lines[1:]
        elif d2 < d1:
            lines = lines[:-1]
        else:
            lines = lines[1:]
    elif ll >= 11:
        log.info("There are 11 or more lines, removing on both sides...")
        lines = lines[1:-1]
    if ll >= 12:
        log.info("There are 12 or more lines, removing extras again...")
        lines = rem_extras(lines, ll-2, k, dd)
    return lines


def length_theta(vert, hori=None, abs_angle=False):
    def _create(lines):
        dummy = np.zeros((lines.shape[0], 7), dtype='int32')
        dummy[:, 0:4] = lines[:, 0:4]
        lines = dummy[np.argsort(dummy[:, 0])]

        for i, line in enumerate(lines):
            x0, y0, x1, y1, r, t, _ = line
            lines[i, 4] = length((x0, y0, x1, y1))
            lines[i, 5] = theta((x0, y0, x1, y1), abs_angle=abs_angle)
        return np.round(lines)

    if hori is not None:
        hori = _create(hori)
    vert = _create(vert)
    return vert, hori


def length(line):
    x0, y0, x1, y1 = line[:4]
    dx = x1 - x0
    dy = y1 - y0
    return np.sqrt(dx*dx + dy*dy)


def theta(line, abs_angle=False):
    x0, y0, x1, y1 = line[:4]
    if abs_angle:
        angle = np.arctan2(abs(y0-y1), abs(x1-x0))
    else:
        if x1 < x0:
            a1, b1 = x0, y0
            x0, y0 = x1, y1
            x1, y1 = a1, b1
        angle = np.arctan2(y0-y1, x1-x0)

    return np.rad2deg(angle)


def calc_intersections(lines0, lines1=None):
    log.info("calculating intersections between group(s) of lines...")

    if lines1 is None:
        lines1 = lines0

    rows = []
    for x0, y0, x1, y1, r, t, _ in lines0:
        col = []
        for xx0, yy0, xx1, yy1, rr, tt, _ in lines1:
            if (x0, y0) == (xx0, yy0) and (x1, y1) == (xx0, yy0):
                continue

            dtheta = abs(t - tt)
            tol0 = consts.min_angle_to_intersect
            tol1 = 180 - tol0
            if (dtheta < tol0 or dtheta > tol1):
                continue

            xdiff = (x0 - x1, xx0 - xx1)
            ydiff = (y0 - y1, yy0 - yy1)

            div = det([xdiff, ydiff])
            if div == 0:
                continue

            d = (det([(x0, y0), (x1, y1)]),
                 det([(xx0, yy0), (xx1, yy1)]))
            x = det([d, xdiff]) / div
            y = det([d, ydiff]) / div
            col.append((x, y))
        rows.append(col)

    inter = np.round(rows)
    return np.array(inter, dtype='int32')


def calc_intersection(line0, ww=500, hh=300, kind=0):
    log.debug("calculating intersection between 2 lines...")
    if kind == 0:
        line1 = (50, 0, 400, 0, 0, 0)
    elif kind == 1:
        line1 = (0, 50, 0, 400, 0, 0)
    elif kind == 2:
        line1 = (50, hh, 400, hh, 0, 0)
    elif kind == 3:
        line1 = (ww, 50, ww, 400, 0, 0)

    x0, y0, x1, y1 = line0[:4]
    xx0, yy0, xx1, yy1 = line1[:4]
    if (x0, y0, x1, x1) == (xx0, yy0, xx1, yy1):
        log.warning("lines should not be equal")
        return (30000, 30000)

    xdiff = (x0 - x1, xx0 - xx1)
    ydiff = (y0 - y1, yy0 - yy1)

    div = det([xdiff, ydiff])
    if div == 0:
        log.warning("div == 0 (parallel lines)")
        return (30000, 30000)

    d = (det([(x0, y0), (x1, y1)]),
         det([(xx0, yy0), (xx1, yy1)]))
    x = round(det([d, xdiff]) / div)
    y = round(det([d, ydiff]) / div)
    return np.array((x, y), dtype='int32')


def check_save(image, title, vert, hori, old_lv=0, old_lh=0):
    lv, lh = len(vert), len(hori)
    if old_lv == lv and old_lh == lh:
        return old_lv, old_lh

    if aux.debugging():
        canvas = draw.lines(image, vert, hori)
        aux.save(title, canvas)
    return lv, lh
