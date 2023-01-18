import numpy as np
import cv2
import logging as log

import auxiliar as aux
import constants as consts


def add_outer(lines, k, ww, hh):
    tol = consts.outer_tolerance
    runs = 0

    while runs < 2:
        lines = calc_outer(lines, tol, 0, k, ww, hh)
        lines = calc_outer(lines, tol, -1, k, ww, hh)
        runs += 1
    return lines


def add_outer_wrap(img, vert, hori):
    log.info("adding missing outer lines...")
    ww = img.bwidth
    hh = img.bheigth

    vert = add_outer(vert, 0, ww, hh)
    hori = add_outer(hori, 1, ww, hh)
    return vert, hori


def add_middle(vert, hori):
    log.info("adding missing middle lines...")
    tol = consts.middle_tolerance

    def _append(lines, i, x, y, kind):
        x1, x2 = x
        y1, y2 = y
        line = (x1, y1, x2, y2)
        new = np.array([[x1, y1, x2, y2,
                         aux.radius(line), aux.theta(line), 0]], dtype='int32')
        lines = np.append(lines, new, axis=0)
        lines, _ = sort_lines(lines, k=kind)
        return lines

    def _add_middle(lines, kind):
        dthis = (abs(lines[0, kind] - lines[1, kind])
                 + abs(lines[0, kind+2] - lines[1, kind+2])) / 2
        dnext = (abs(lines[1, kind] - lines[2, kind])
                 + abs(lines[1, kind+2] - lines[2, kind+2])) / 2
        dnext2 = (abs(lines[2, kind] - lines[3, kind])
                  + abs(lines[2, kind+2] - lines[3, kind+2])) / 2
        x = (lines[1, 0] - lines[2, 0] + lines[1, 0],
             lines[1, 2] - lines[2, 2] + lines[1, 2])
        y = (lines[1, 1] - lines[2, 1] + lines[1, 1],
             lines[1, 3] - lines[2, 3] + lines[1, 3])
        if dthis > (dnext*tol) and dthis > (dnext2*tol):
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


def add_last_outer(vert, hori):
    return vert, hori


def split_lines(lines):
    log.info("spliting lines into vertical and horizontal...")
    if (lines.shape[1] < 6):
        lines, _ = aux.radius_theta(lines)
    lines = np.array(lines, dtype='float32')

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compact, labels, centers = cv2.kmeans(lines[:, 5], 3, None,
                                          criteria, 15, flags)
    labels = np.ravel(labels)

    d1 = abs(centers[0] - centers[1])
    d2 = abs(centers[0] - centers[2])
    d3 = abs(centers[1] - centers[2])

    maxdiff = consts.angles_max_diff
    dd1 = d1 < maxdiff and d2 > maxdiff and d3 > maxdiff
    dd2 = d2 < maxdiff and d1 > maxdiff and d3 > maxdiff
    dd3 = d3 < maxdiff and d1 > maxdiff and d2 > maxdiff

    if dd1 or dd2 or dd3:
        compact, labels, centers = cv2.kmeans(lines[:, 5], 2, None,
                                              criteria, 15, flags)
        labels = np.ravel(labels)
        A = lines[labels == 0]
        B = lines[labels == 1]
    else:
        # redo kmeans using absolute inclination
        lines, _ = aux.radius_theta(lines, abs_angle=True)
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
        vert = np.int32(B, dtype='int32')
        hori = np.int32(A, dtype='int32')
    nvert = []
    for line in vert:
        x1, y1, x2, y2 = line[:4]
        if y1 > y2:
            a1, b1 = x1, y1
            x1, y1 = x2, y2
            x2, y2 = a1, b1
        line = [x1, y1, x2, y2, line[4], line[5]]
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
            inter = aux.calc_intersection(line, kind=kind)
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
        x1 = line0[0] + dx[0]
        y1 = line0[1] + dy[0]
        x2 = line0[2] + dx[1]
        y2 = line0[3] + dy[1]
        new = np.array([[x1, y1, x2, y2, 0, 0, 0]], dtype='int32')
        inters = limit_bydims(new[0][:4], ww, hh)
        if len(inters) != 2:
            return lines
        if inters[0, k] <= dd and inters[1, k] <= dd:
            x1, y1, x2, y2 = np.ravel(inters)
            line = (x1, y1, x2, y2)
            minlen = consts.min_line_length / 1.5
            if (r := aux.radius(line)) >= minlen:
                new = np.array([[x1, y1, x2, y2,
                                 r, aux.theta(line), 0]], dtype='int32')
                lines = np.append(lines, new, axis=0)
                lines, _ = sort_lines(lines, k=k)
    return lines


def calc_inters(line, ww, hh):
    i0 = aux.calc_intersection(line, ww, hh, kind=0)
    i1 = aux.calc_intersection(line, ww, hh, kind=1)
    i2 = aux.calc_intersection(line, ww, hh, kind=2)
    i3 = aux.calc_intersection(line, ww, hh, kind=3)
    return np.array([i0, i1, i2, i3], dtype='int32')


def shorten_byinter(img, ww, hh, vert, hori=None):
    inters = aux.calc_intersections(img.gray3ch, vert, hori)

    def _shorten(lines):
        nlines = []
        for i, inter in enumerate(inters):
            line = lines[i]
            a, b = inter[0], inter[-1]
            new = np.array([[a[0], a[1], b[0], b[1], 0, 0, 0]], dtype='int32')
            limit = limit_bydims(new[0, :4], ww, hh)
            limit = np.ravel(limit)
            if aux.radius(limit) < aux.radius(new[0, :4]):
                x1, y1, x2, y2 = limit[:4]
            else:
                x1, y1, x2, y2 = new[0, :4]
            new = x1, y1, x2, y2, line[4], line[5], line[6]
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
