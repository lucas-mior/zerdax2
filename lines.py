import numpy as np
from numpy.linalg import det
import cv2
import logging as log
from jenkspy import jenks_breaks

import algorithm as algo
import constants as consts
import drawings as draw
from c_load import segments_distance
from c_load import lines_bundle

minlen0 = consts.min_line_length
canny3ch = None
WLEN = 512


def find_lines(canny):
    log.info("finding all lines of board...")
    ww = canny.shape[1]
    hh = canny.shape[0]

    vert, hori = find_baselines(canny)
    vert, hori = fix_length_byinter(ww, hh, vert, hori)
    lv, lh = check_save("fix_length_byinter0", vert, hori, -1, -1)

    if lv == 0 or lh == 0:
        return None, None

    vert, hori = expand_borders(ww, hh, vert, hori)
    inters = calc_intersections(vert, hori)
    corners = calc_corners(inters)
    print(corners)

    vert = fix_lines(vert, 0, ww, hh)
    hori = fix_lines(hori, 1, ww, hh)
    vert, hori = fix_length_byinter(ww, hh, vert, hori)
    lv, lh = check_save("fix_length_byinter1", vert, hori, -1, -1)
    vert = fix_lines(vert, 0, ww, hh)
    hori = fix_lines(hori, 1, ww, hh)

    return vert, hori

def expand_borders(ww, hh, vert, hori):

    def _expand(lines, dim):
        return lines;

    vert = _expand(vert, ww)
    hori = _expand(hori, hh)
    return vert, hori


def calc_corners(inters):
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

    return np.array([BR, BL, TR, TL], dtype='int32')


def fix_lines(lines, kind, ww, hh):
    ll = len(lines)
    l2 = 0
    runs = 0
    while ll != l2 and runs <= 5:
        ll = l2
        lines, l2 = rem_outer(lines, l2, kind, ww, hh)
        if len(lines) >= 7:
            lines, l2 = rem_wrong(lines, l2, kind, ww, hh)
        if len(lines) <= 9:
            lines, l2 = add_outer(lines, l2, kind, ww, hh, force=True)
        else:
            lines, l2 = add_outer(lines, l2, kind, ww, hh)
        if len(lines) >= 7:
            lines, l2 = rem_middle(lines, l2, kind, ww, hh)
        lines, l2 = add_middle(lines, l2, kind, ww, hh)
        lines, l2 = add_middle(lines, l2, kind, ww, hh)
        if (l2 >= 10):
            lines, l2 = rem_outer(lines, l2, kind, ww, hh, force=True)
        runs += 1

    return lines


def find_baselines(canny):
    ww = canny.shape[1]
    hh = canny.shape[0]
    distv = round(ww/23)
    disth = round(hh/23)
    global canny3ch
    canny3ch = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    min_before_split = consts.min_lines_before_split

    angle = consts.hough_angle_resolution
    tangle = np.deg2rad(angle)
    minlen = minlen0
    maxgap = 4
    tvotes = round(minlen0*1.1)
    lv = lh = 0
    hori = vert = None
    while (lv < 8 or lh < 8):
        if tvotes <= round(minlen0/1.5) and (minlen <= round(minlen0/1.1)):
            break
        minlen = max(minlen - 5, round(minlen0 / 1.1))
        maxgap = min(maxgap + 2, round(minlen0 / 4))
        tvotes = max(tvotes - 5, round(minlen0 / 1.5))
        lines = cv2.HoughLinesP(canny, 1, tangle, tvotes, None, minlen, maxgap)
        if lines is None:
            log.debug(f"0 @ {angle}, {tvotes=}, {minlen=}, {maxgap=}")
            maxgap += 5
            continue
        elif (ll := lines.shape[0]) < min_before_split:
            log.debug(f"{ll} @ {angle}, {tvotes=}, {minlen=}, {maxgap=}")
            maxgap += 2
            continue

        lines_hough = lines[:, 0, :]
        if algo.debug:
            canvas = draw.lines(canny3ch, lines_hough)
            ll = len(lines_hough)
            log.debug(f"{ll} @ {angle}, {tvotes=}, {minlen=}, {maxgap=}")
            draw.save("hough_lines", canvas)

        lines_hough, _ = length_theta(lines_hough, abs_angle=False)
        vert, hori = split_lines(lines_hough)
        lv, lh = check_save("split_lines", vert, hori, 0, 0)
        vert, hori = filter_byangle(vert, hori)
        lv, lh = check_save("filter_byangle", vert, hori, lv, lh)
        vert, hori = sort_lines(vert, hori)
        lv, lh = check_save("sort_lines", vert, hori, 0, 0)
        if vert is None or hori is None:
            continue
        bundled = np.zeros(vert.shape, dtype='int32')
        nlines = lines_bundle(vert, bundled, len(vert), distv)
        vert = bundled[:nlines]
        bundled = np.zeros(hori.shape, dtype='int32')
        nlines = lines_bundle(hori, bundled, len(hori), disth)
        hori = bundled[:nlines]
        lv, lh = check_save("lines_bundled", vert, hori, lv, lh)
        vert, hori = filter_byinter(vert, hori)
        lv, lh = check_save("filter_byinter", vert, hori, lv, lh)

        ll = lv + lh
        log.info(f"{ll} # {lv},{lh} @ {angle}º, {tvotes=},{minlen=},{maxgap=}")

    if lv != 9 or lh != 9:
        log.warning("Wrong lines found in at least one direction")
        # canvas = draw.lines(canny3ch, vert, hori)
        # draw.save(f"canny{lv=}_{lh=}", canvas)
    if lv < 6 or lh < 6:
        log.error("Less than 6 lines found in at least one direction")
        canvas = draw.lines(canny3ch, vert, hori)
        draw.save(f"canny{lv=}_{lh=}", canvas)
        return None, None

    vert, hori = sort_lines(vert, hori)
    lv, lh = check_save("sort_lines", vert, hori, 0, 0)
    return vert, hori


def split_lines(lines):
    log.info("spliting lines into vertical and horizontal...")

    def _check_split(vert, hori):
        if abs(np.median(hori[:, 5]) - np.median(vert[:, 5])) < 40*100:
            return False
        else:
            return True

    if (lines.shape[1] < 6):
        lines, _ = length_theta(lines)
    angles = lines[:, 5]

    try:
        limits = jenks_breaks(angles, n_classes=3)
    except Exception:
        return None, None

    a0 = angles[angles <= limits[1]]
    a1 = angles[(limits[1] < angles) & (angles <= limits[2])]
    a2 = angles[limits[2] < angles]
    centers = [np.median(a0), np.median(a1), np.median(a2)]

    d0 = abs(centers[0] - centers[1])
    d1 = abs(centers[0] - centers[2])
    d2 = abs(centers[1] - centers[2])

    maxdiff = consts.angles_max_diff
    dd0 = d0 < maxdiff and d1 > maxdiff and d2 > maxdiff
    dd1 = d1 < maxdiff and d0 > maxdiff and d2 > maxdiff
    dd2 = d2 < maxdiff and d0 > maxdiff and d1 > maxdiff

    if dd0 or dd1 or dd2:
        try:
            limits = jenks_breaks(angles, n_classes=2)
        except Exception:
            return None, None
        hori = lines[angles <= limits[1]]
        vert = lines[limits[1] < angles]
        if not _check_split(vert, hori):
            return None, None
    else:
        for line in lines:
            if line[5] < (-45 * 100):
                line[5] = -line[5]
        angles = lines[:, 5]
        try:
            limits = jenks_breaks(angles, n_classes=2)
        except Exception:
            return None, None
        hori = lines[angles <= limits[1]]
        vert = lines[limits[1] < angles]
        if not _check_split(vert, hori):
            return None, None

    if abs(np.median(vert[:, 5])) < abs(np.median(hori[:, 5])):
        aux = vert
        vert = hori
        hori = aux

    for line in vert:
        if line[1] > line[3]:
            a1, b1 = line[0], line[1]
            line[0], line[1] = line[2], line[3]
            line[2], line[3] = a1, b1
    return vert, hori


def filter_byangle(vert, hori=None):
    log.info("filtering lines by angle accoring to direction...")
    tol = consts.angle_tolerance

    def _filter(lines):
        angle = np.median(lines[:, 5])
        right = np.abs(lines[:, 5] - angle) <= tol
        return lines[right]

    if vert is not None:
        vert = _filter(vert)
    if hori is not None:
        hori = _filter(hori)
    return vert, hori


def filter_byinter(vert, hori=None):
    log.info("filtering lines by number of intersections ...")

    def _filter(lines):
        for i, line in enumerate(lines):
            inters = calc_intersections(np.array([line]), lines, limit=True)
            if inters is None or len(inters) == 0:
                continue
            elif inters.shape[1] >= 4:
                lines = np.delete(lines, i, axis=0)
                break
        return lines

    for j in range(3):
        if vert is not None:
            vert = _filter(vert)
        if hori is not None:
            hori = _filter(hori)
    return vert, hori


def sort_lines(vert, hori=None, k=0):
    log.debug("sorting lines by position and respective direction...")

    def _create(lines, kind):
        positions = np.empty(lines.shape[0], dtype='int32')
        for i, line in enumerate(lines):
            inter = calc_intersection(line, kind=kind)
            positions[i] = inter[kind]
        return positions

    if hori is not None:
        positions = _create(hori, kind=1)
        hori = hori[np.argsort(positions)]
        if vert is not None:
            positions = _create(vert, kind=0)
            vert = vert[np.argsort(positions)]
    else:
        if vert is not None:
            positions = _create(vert, kind=k)
            vert = vert[np.argsort(positions)]
    return vert, hori


def fix_length_byinter(ww, hh, vert, hori=None):
    inters = calc_extern_intersections(vert, hori)
    if inters is None:
        log.debug("fix_length by inter did not find any intersection")
        return vert, hori

    def _shorten(lines):
        for i, inter in enumerate(inters):
            line = lines[i]
            a, b = inter[0], inter[-1]
            new = np.array([a[0], a[1], b[0], b[1]], dtype='int32')
            limit = limit_bydims(new, ww, hh)
            limit = np.ravel(limit[:2])
            if length(limit) < length(new):
                x0, y0, x1, y1 = limit
            else:
                x0, y0, x1, y1 = new
            new = x0, y0, x1, y1, line[4], line[5]
            lines[i] = new
        return lines

    if vert is not None:
        vert = _shorten(vert)
    if hori is not None:
        inters = np.transpose(inters, axes=(1, 0, 2))
        hori = _shorten(hori)
    return vert, hori


def check_save(title, vert, hori, old_lv, old_lh):
    global canny3ch
    number = True
    lv = 0 if vert is None else len(vert)
    lh = 0 if hori is None else len(hori)

    if old_lv == lv and old_lh == lh:
        return old_lv, old_lh

    if algo.debug:
        if lv > 12 or lh > 12:
            number = False
        canvas = draw.lines(canny3ch, vert, hori, number=number)
        draw.save(title, canvas)
    return lv, lh


def add_outer(lines, ll, k, ww, hh, force=False):
    log.info("adding missing outer lines...")
    tol = consts.outer_tolerance
    if force:
        tol = 1

    def _add_outer(lines, tol, where, ww, hh):
        dd = ww if k == 0 else hh
        if where == 0:
            ee = 0
            other = 1
        elif where == -1:
            ee = dd
            other = -2

        line0 = lines[where]
        line1 = lines[other]

        x = (2*line0[0] - line1[0], 2*line0[2] - line1[2])
        y = (2*line0[1] - line1[1], 2*line0[3] - line1[3])
        if abs(line0[k] - ee) > tol and abs(line0[k+2] - ee) > tol:
            x0, x1 = x
            y0, y1 = y
            new = np.array([x0, y0, x1, y1], dtype='int32')
            inters = limit_bydims(new, ww, hh)
            if len(inters) < 2:
                return lines
            elif len(inters) > 2:
                ls = np.array([[inters[0], inters[1]],
                               [inters[0], inters[2]],
                               [inters[1], inters[2]]])
                lens = [length(le.flatten()) for le in ls]
                inters = ls[np.argmax(lens)]
            if inters[0, k] <= dd and inters[1, k] <= dd:
                x0, y0, x1, y1 = np.ravel(inters)
                if length((x0, y0, x1, y1)) < (length(line0)*0.8):
                    return lines
                new = np.array([[x0, y0, x1, y1,
                                 line1[4], line1[5]]], dtype='int32')
                if where == -1:
                    lines = np.append(lines, new, axis=0)
                else:
                    lines = np.insert(lines, 0, new, axis=0)
        return lines

    lines = _add_outer(lines, tol, 0, ww, hh)
    ll, _ = check_save("add_outer0", lines, None, ll, 0)
    lines = _add_outer(lines, tol, -1, ww, hh)
    ll, _ = check_save("add_outer1", lines, None, ll, 0)
    return lines, ll


def rem_middle(lines, ll, kind, ww, hh, force=False):
    log.info("reming missing middle lines...")
    tol = consts.middle_tolerance

    def _rem_middle(lines):
        i = 1
        dprev1 = segments_distance(lines[i+0], lines[i-1])
        dprev0 = segments_distance(lines[i+0], lines[i+1])
        dnext0 = segments_distance(lines[i+1], lines[i+2])
        dnext1 = segments_distance(lines[i+2], lines[i+3])
        dnext2 = segments_distance(lines[i+3], lines[i+4])
        if dprev1 < (dnext0/tol) > dprev0 and dprev1 < (dnext1/tol) > dprev0:
            if dprev1 < (dnext2/tol) > dprev0:
                lines = np.delete(lines, i, axis=0)
                return lines
        for i in range(3, len(lines) - 3):
            dprev2 = segments_distance(lines[i-2], lines[i-3])
            dprev1 = segments_distance(lines[i-1], lines[i-2])
            dprev0 = segments_distance(lines[i+0], lines[i-1])
            dnext0 = segments_distance(lines[i+0], lines[i+1])
            dnext1 = segments_distance(lines[i+1], lines[i+2])
            dnext2 = segments_distance(lines[i+2], lines[i+3])
            if dprev0 < (dprev1/tol) and dnext0 < (dnext1/tol):
                if dprev0 < (dprev2/tol) and dnext0 < (dnext2/tol):
                    lines = np.delete(lines, i, axis=0)
                    return lines
        return lines

    lines = _rem_middle(lines)
    lines = np.flip(lines, axis=0)
    lines = _rem_middle(lines)
    lines = np.flip(lines, axis=0)
    ll, _ = check_save("rem_middle", lines, None, ll, 0)
    return lines, ll


def rem_wrong(lines, ll, kind, ww, hh, force=False):
    log.info("removing wrong middle lines...")
    tol = consts.middle_tolerance

    def _calc_dists(lines):
        dists = np.zeros((lines.shape[0], 2), dtype='int32')
        i = 0
        dists[i, 0] = segments_distance(lines[i+0], lines[i+1])
        dists[i, 1] = segments_distance(lines[i+0], lines[i+1])
        for i in range(1, len(lines) - 1):
            dists[i, 0] = segments_distance(lines[i+0], lines[i-1])
            dists[i, 1] = segments_distance(lines[i+0], lines[i+1])
        i += 1
        dists[i, 0] = segments_distance(lines[i+0], lines[i-1])
        dists[i, 1] = segments_distance(lines[i+0], lines[i-1])
        return dists

    def _rem_wrong(lines, dists):
        d0 = np.median(dists[:, 0])
        d1 = np.median(dists[:, 1])
        med = round((d0 + d1)/2)
        for i in range(1, len(lines) - 1):
            if dists[i, 1] < (med/(tol+0.1)) and (med*tol) < dists[i, 0]:
                if (dists[i-1, 0]/tol) > dists[i, 1]:
                    lines = np.delete(lines, i, axis=0)
                elif (dists[i+1, 1]*tol) < dists[i, 0]:
                    lines = np.delete(lines, i, axis=0)
                return lines
        return lines

    dists = _calc_dists(lines)
    lines = _rem_wrong(lines, dists)
    lines = np.flip(lines, axis=0)
    dists = np.flip(dists, axis=0)
    dists = np.flip(dists, axis=-1)
    lines = _rem_wrong(lines, dists)
    lines = np.flip(lines, axis=0)
    ll, _ = check_save("rem_wrong", lines, None, ll, 0)
    return lines, ll


def add_middle(lines, ll, kind, ww, hh, force=False):
    log.info("adding missing middle lines...")
    tol = consts.middle_tolerance

    def _insert(lines, i, x, y):
        x0, x1 = x
        y0, y1 = y
        new = np.array([[x0, y0, x1, y1,
                         lines[i-1, 4], lines[i-1, 5]]], dtype='int32')
        lines = np.insert(lines, i+1, new, axis=0)
        return lines

    def _add_middle(lines):
        dnext0 = segments_distance(lines[0], lines[1])
        dnext1 = segments_distance(lines[1], lines[2])
        dnext2 = segments_distance(lines[2], lines[3])
        dnext3 = segments_distance(lines[3], lines[4])
        if dnext0 > (dnext1*tol) and dnext0 > (dnext2*(tol-0.05)):
            if dnext0 > (dnext3*(tol-0.10)):
                x = (2*lines[1, 0] - lines[2, 0],
                     2*lines[1, 2] - lines[2, 2])
                y = (2*lines[1, 1] - lines[2, 1],
                     2*lines[1, 3] - lines[2, 3])
                lines = _insert(lines, 0, x, y)
                if segments_distance(lines[0], lines[1]) < (dnext0/(tol+0.9)):
                    lines = lines[1:]
                return lines
        for i in range(2, len(lines) - 3):
            dprev2 = segments_distance(lines[i-1], lines[i-2])
            dprev1 = segments_distance(lines[i+0], lines[i-1])
            dthis = segments_distance(lines[i+0], lines[i+1])
            dnext1 = segments_distance(lines[i+1], lines[i+2])
            dnext2 = segments_distance(lines[i+2], lines[i+3])
            if dthis > (dprev1*tol) and dthis > (dnext1*tol):
                if dthis > (dprev2*(tol-0.05)) and dthis > (dnext2*(tol-0.05)):
                    if dthis > (dprev1*3):
                        x = (2*lines[i, 0] - lines[i-1, 0],
                             2*lines[i, 2] - lines[i-1, 2])
                        y = (2*lines[i, 1] - lines[i-1, 1],
                             2*lines[i, 3] - lines[i-1, 3])
                    else:
                        x = (round((lines[i, 0] + lines[i+1, 0])/2),
                             round((lines[i, 2] + lines[i+1, 2])/2))
                        y = (round((lines[i, 1] + lines[i+1, 1])/2),
                             round((lines[i, 3] + lines[i+1, 3])/2))
                    lines = _insert(lines, i, x, y)
                    return lines
        for i in range(1, len(lines) - 4):
            dprev0 = segments_distance(lines[i+0], lines[i-1])
            dnext0 = segments_distance(lines[i+0], lines[i+1])
            dnext1 = segments_distance(lines[i+1], lines[i+2])
            dnext2 = segments_distance(lines[i+2], lines[i+3])
            dnext3 = segments_distance(lines[i+3], lines[i+4])
            if dnext0 > (dnext1*tol) and dnext0 > (dnext2*(tol-0.05)):
                if dnext0 > (dprev0*(tol-0.1)) or dnext0 > (dnext3*(tol-0.1)):
                    x = (round((lines[i, 0] + lines[i+1, 0])/2),
                         round((lines[i, 2] + lines[i+1, 2])/2))
                    y = (round((lines[i, 1] + lines[i+1, 1])/2),
                         round((lines[i, 3] + lines[i+1, 3])/2))
                    lines = _insert(lines, i, x, y)
                    return lines
        return lines

    lines = _add_middle(lines)
    lines = np.flip(lines, axis=0)
    ll, _ = check_save("add_middle0", lines, None, ll, 0)
    lines = _add_middle(lines)
    lines = np.flip(lines, axis=0)
    ll, _ = check_save("add_middle1", lines, None, ll, 0)
    return lines, ll


def rem_outer(lines, ll, k, ww, hh, force=False):
    log.info("removing extra outer lines...")
    tol = consts.outer_tolerance
    if k == 0:
        dd = ww
    else:
        dd = hh

    d00 = abs(lines[1, k] - 0)
    d01 = abs(lines[1, k+2] - 0)
    d0 = min(d00, d01)
    d10 = abs(lines[-2, k] - dd)
    d11 = abs(lines[-2, k+2] - dd)
    d1 = min(d10, d11)
    if not force:
        if d0 < tol:
            lines = lines[1:]
        if d1 < tol:
            lines = lines[:-1]
    else:
        if d0 < d1:
            lines = lines[1:]
        else:
            lines = lines[:-1]

    ll, _ = check_save("rem_outer", lines, None, ll, 0)
    return lines, ll


def limit_bydims(inters, ww, hh):
    i0 = calc_intersection(inters, ww, hh, kind=0)
    i1 = calc_intersection(inters, ww, hh, kind=1)
    i2 = calc_intersection(inters, ww, hh, kind=2)
    i3 = calc_intersection(inters, ww, hh, kind=3)
    inters = np.array([i0, i1, i2, i3], dtype='int32')

    inters = inters[(inters[:, 0] >= 0) & (inters[:, 1] >= 0) &
                    (inters[:, 0] <= ww) & (inters[:, 1] <= hh)]
    return inters


def length_theta(vert, hori=None, abs_angle=False):
    if abs_angle:
        angle = theta_abs
    else:
        angle = theta

    def _create(lines):
        dummy = np.empty((lines.shape[0], 6), dtype='int32')
        dummy[:, 0:4] = lines[:, 0:4]
        lines = dummy[np.argsort(dummy[:, 0])]

        for i, line in enumerate(lines):
            x0, y0, x1, y1, r, t = line[:6]
            lines[i, 4] = length((x0, y0, x1, y1))
            lines[i, 5] = angle((x0, y0, x1, y1))*100
        return lines

    if hori is not None:
        hori = _create(hori)
    vert = _create(vert)
    return vert, hori


def length(line):
    x0, y0, x1, y1 = line[:4]
    dx = x1 - x0
    dy = y1 - y0
    return np.sqrt(dx*dx + dy*dy)


def theta(line):
    x0, y0, x1, y1 = line[:4]
    if x1 < x0:
        a1, b1 = x0, y0
        x0, y0 = x1, y1
        x1, y1 = a1, b1
    angle = np.arctan2(y0-y1, x1-x0)
    return np.rad2deg(angle)


def theta_abs(line):
    x0, y0, x1, y1 = line[:4]
    angle = np.arctan2(abs(y0-y1), abs(x1-x0))
    return np.rad2deg(angle)


def calc_extern_intersections(lines0, lines1=None):
    log.info("calculating external intersections between group(s) of lines...")

    if lines1 is None:
        lines1 = lines0
    if lines0 is None:
        return None

    rows = []
    for i in range(l0 := lines0.shape[0]):
        x0, y0, x1, y1, r, t = lines0[i]
        col = []
        for j in range(l1 := lines1.shape[0]):
            xx0, yy0, xx1, yy1, rr, tt = lines1[j]
            if 0 != i != (l0-1) and 0 != j != (l1-1):
                col.append((30000, 30000))
                continue
            if (x0, y0, x1, x1) == (xx0, yy0, xx1, yy1):
                continue

            dtheta = abs(t - tt)
            tol0 = consts.min_angle_to_intersect
            tol1 = 180*100 - tol0
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

    try:
        inter = np.round(rows)
        return np.array(inter, dtype='int32')
    except Exception:
        return None


def calc_intersections(lines0, lines1=None, onlylast=False, limit=False):
    if lines1 is None:
        lines1 = lines0

    maxx0 = np.max([lines0[:, 0], lines0[:, 2]])
    maxx1 = np.max([lines1[:, 0], lines1[:, 2]])
    maxy0 = np.max([lines0[:, 1], lines0[:, 3]])
    maxy1 = np.max([lines1[:, 1], lines1[:, 3]])
    maxx = max(maxx0, maxx1)
    maxy = max(maxy0, maxy1)

    rows = []
    for line0 in lines0:
        x0, y0, x1, y1, r, t = line0
        col = []
        for line1 in lines1:
            xx0, yy0, xx1, yy1, rr, tt = line1
            if (x0, y0, x1, x1) == (xx0, yy0, xx1, yy1):
                continue

            if not limit:
                dtheta = abs(t - tt)
                tol0 = consts.min_angle_to_intersect
                tol1 = 180*100 - tol0
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
            if limit and (x < 0 or x > maxx or y < 0 or y > maxy):
                continue
            col.append((x, y))
        rows.append(col)

    inter = np.round(rows)
    return np.array(inter, dtype='int32')


def calc_intersection(line0, ww=500, hh=300, kind=0):
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


def find_corners(img):
    # img = create_cannys(img)
    # vert, hori = magic_lines(img)
    # inters = aux.calc_intersections(img.gray3ch, vert, hori)
    # canvas = draw.intersections(img.gray3ch, inters)
    # aux.save(img, "intersections", canvas)
    inters = None

    # img.corners = calc_corners(img, inters)

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
    draw.save(img, canvas, "A1E4C5H8")
    squares = np.float32(squares)
    # scale to input size
    squares[:, :, :, 0] /= img.bfact
    squares[:, :, :, 1] /= img.bfact
    # position board bounding box
    squares[:, :, :, 0] += img.x0
    squares[:, :, :, 1] += img.y0

    img.squares = np.array(np.round(squares), dtype='int32')
    return img


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
    draw.save("warpclaheG", img.wg)
    draw.save("warpclaheV", img.wv)

    return img


def find_squares(img):
    print("generating 3 channel gray warp image for drawings...")
    img.warp3ch = cv2.cvtColor(img.wg, cv2.COLOR_GRAY2BGR)

    inter = calc_intersections(vert, hori)
    inter = inter.reshape((-1, 2))
    canvas = draw.intersections(img.warp3ch, [inter])
    draw.save(img, canvas, "intersections")
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
    draw.save(img, canvas, "A1E4C5H8")

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


def filter_90(lines):
    rem = np.zeros(lines.shape[0], dtype='uint8')

    for i, t in enumerate(lines[:, 5]):
        if abs(t - 90) > 4 and abs(t + 90) > 4 and abs(t) > 4:
            rem[i] = 1
        else:
            rem[i] = 0

    return lines[rem == 0]


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
