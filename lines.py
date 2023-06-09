import numpy as np
import cv2
import logging as log
from jenkspy import jenks_breaks

import algorithm
import intersect
import draw
from c_load import segments_distance
from c_load import lines_bundle

gcanny = None


def find_warped_lines(canny):
    global gcanny
    gcanny = canny
    log.debug("finding right angled lines of warped board...")

    min_lines_before_split = 16
    hough_min_length0 = 0.9*gcanny.shape[1]
    hough_max_gap = 8

    angle = 0.5
    hough_min_length = hough_min_length0
    hough_threshold = hough_min_length0

    ll = lh = lv = 0
    hori = vert = None
    while lh < 7 or lv < 7:
        if hough_threshold <= (hough_min_length0/1.7):
            if hough_min_length <= (hough_min_length0/1.1):
                break
        hough_min_length = max(hough_min_length - 3, hough_min_length0 / 1.1)
        hough_max_gap = min(hough_max_gap + 3, hough_min_length0 / 4)
        hough_threshold = max(hough_threshold - 8, hough_min_length0 / 1.7)

        lines, ll = hough(hough_threshold, hough_min_length, hough_max_gap)
        if ll < min_lines_before_split:
            continue

        lines = filter_not_right(lines)
        hori, vert, lh, lv = split(lines)
        if lh < 4 or lv < 4:
            continue

        hori, vert = sort(hori, vert)
        hori, vert = bundle_lines(hori, vert)

        lh, lv = len(hori), len(vert)
        log.debug(f"{lv+lh} # {lv},{lh} @ {angle}º, {hough_threshold=}, "
                  f"{hough_min_length=},{hough_max_gap=}")

    if (failed := lv < 6 or lh < 6) or algorithm.debug:
        canvas = draw.lines(gcanny, hori, vert)
        draw.save(f"warped_lines_{lv=}_{lh=}", canvas)
        if failed:
            log.error("less than 6 lines found in at least one direction")
            return None, None

    hori, vert = sort(hori, vert)
    hori, vert = fix_warped_lines(hori, vert)
    return hori, vert


def find_diagonal_lines(canny):
    global gcanny
    gcanny = canny
    log.debug("finding diagonal lines of original board...")

    min_lines_before_split = 16
    hough_min_length0 = gcanny.shape[0] * 0.7
    hough_max_gap = 5

    angle = 0.5
    hough_min_length = hough_min_length0
    hough_threshold = hough_min_length0

    ll = lh = lv = 0
    hori = vert = None
    while lv < 7 or lh < 7:
        if hough_threshold <= (hough_min_length0/1.5):
            if hough_min_length <= (hough_min_length0/1.5):
                break
        hough_min_length = max(hough_min_length - 5, hough_min_length0 / 1.5)
        hough_max_gap = min(hough_max_gap + 3, hough_min_length0 / 4)
        hough_threshold = max(hough_threshold - 2, hough_min_length0 / 1.5)

        lines, ll = hough(hough_threshold, hough_min_length, hough_max_gap)
        if ll < min_lines_before_split:
            continue

        hori, vert, lh, lv = split(lines)
        if lv < 4 or lh < 4:
            continue

        hori, vert = filter_misdirected(hori, vert)
        hori, vert = sort(hori, vert)
        # hori, vert = filter_misdirected2(hori, vert)
        hori, vert = bundle_lines(hori, vert)
        hori, vert = filter_intersecting(hori, vert)

        lh, lv = len(hori), len(vert)
        log.debug(f"{lv+lh} # {lh},{lv} @ {angle}º, {hough_threshold=}, "
                  f"{hough_min_length=},{hough_max_gap=}")

    if (failed := lh < 6 or lv < 6) or algorithm.debug:
        canvas = draw.lines(gcanny, hori, vert)
        draw.save(f"diagonal_lines_{lh=}_{lv=}", canvas)
        if failed:
            log.error("less than 6 lines found in at least one direction")
            return None, None

    hori, vert = sort(hori, vert)
    hori, vert = fix_diagonal_lines(hori, vert)
    return hori, vert


def hough(hough_threshold, hough_min_length, hough_max_gap):
    angle = 0.5
    hough_angle = np.deg2rad(angle)
    hough_min_length = round(hough_min_length)
    hough_max_gap = round(hough_max_gap)
    hough_threshold = round(hough_threshold)
    lines = cv2.HoughLinesP(gcanny, 1, hough_angle, hough_threshold,
                            None, hough_min_length, hough_max_gap)
    if lines is None:
        ll = 0
    else:
        lines = lines[:, 0, :]
        ll = len(lines)

    log.debug(f"{ll} @ {angle}, {hough_threshold=}, "
              f"{hough_min_length=}, {hough_max_gap=}")
    if algorithm.debug:
        canvas = draw.lines(gcanny, lines)
        draw.save("hough_lines", canvas)
    return lines, ll


def bundle_lines(hori, vert):
    dist_hori = round(gcanny.shape[0]/23)
    dist_vert = round(gcanny.shape[1]/23)
    old_lh = len(hori)
    old_lv = len(vert)

    def _bundle_lines(lines, dist):
        bundled = np.zeros(lines.shape, dtype='int32')
        nlines = lines_bundle(lines, bundled, len(lines), dist)
        lines = bundled[:nlines]
        return lines, nlines

    hori, lh = _bundle_lines(hori, dist_hori)
    vert, lv = _bundle_lines(vert, dist_vert)

    if algorithm.debug and (old_lh != lh or old_lv != lv):
        canvas = draw.lines(gcanny, hori, vert, annotate_number=True)
        draw.save("bundle_lines", canvas)
    return hori, vert


def fix_warped_lines(hori, vert):

    def _fix_warped_lines(lines, kind):
        lines, ll = remove_wrong(lines, len(vert), kind)
        old_ll = 0
        while old_ll != len(lines):
            old_ll = len(lines)
            lines, ll = add_middle_warped(lines, ll, kind)
        lines, ll = add_outer_warped(lines, ll, kind)
        lines, ll = remove_outer(lines, ll, kind)
        return lines, ll

    def _fix_outer(lines, ll, kind):
        old_ll = 0
        while ll < 11 and old_ll != len(lines):
            old_ll = len(lines)
            lines, ll = add_outer_warped(lines, ll, kind)
        old_ll = 0
        while ll > 9 and old_ll != len(lines):
            old_ll = len(lines)
            lines, ll = remove_outer(lines, ll, kind)
        return lines, ll

    hori, lh = _fix_warped_lines(hori, 1)
    vert, lv = _fix_warped_lines(vert, 0)
    hori, lh = _fix_outer(hori, lh, 1)
    vert, lv = _fix_outer(vert, lv, 0)

    if (failed := lv != 9 or lh != 9) or algorithm.debug:
        canvas = draw.lines(gcanny, hori, vert, annotate_number=True)
        draw.save(f"fix_warped_{lh=}_{lv=}", canvas)
        if failed:
            return None, None
    return hori, vert


def fix_diagonal_lines(hori, vert):
    old_lv = old_lh = 0
    while old_lv != len(vert) or old_lh != len(hori):
        old_lv, old_lh = len(vert), len(hori)
        hori, vert = fix_length_byinter(hori, vert)
        vert, _ = add_outer_diagonal(vert, len(vert), 0)
        hori, vert = fix_length_byinter(hori, vert)
        hori, _ = add_outer_diagonal(hori, len(hori), 1)

    vert, lv = remove_fake_outer(vert, len(vert), 0)
    hori, lh = remove_fake_outer(hori, len(hori), 1)
    vert, lv = extend_outer(vert, len(vert), 0)
    hori, lh = extend_outer(hori, len(hori), 1)
    return hori, vert


def sort(hori, vert):
    log.debug("sorting lines by position and respective direction...")

    def _criteria(lines, kind):
        positions = np.empty(lines.shape[0], dtype='int32')
        for i, line in enumerate(lines):
            inter = intersect.calculate_single(line, gcanny, kind)
            positions[i] = inter[kind]
        return positions

    positions = _criteria(hori, 1)
    hori = hori[np.argsort(positions)]
    positions = _criteria(vert, 0)
    vert = vert[np.argsort(positions)]

    if algorithm.debug:
        canvas = draw.lines(gcanny, hori, vert, annotate_number=True)
        draw.save("sort_lines", canvas)
    return hori, vert


def fix_length_byinter(hori, vert=None):
    log.debug("fixing lines lengths by intersections with other direction...")
    inters = intersect.calculate_extern(hori, vert)

    def _shorten(lines):
        for i, inter in enumerate(inters):
            new = np.ravel([inter[0], inter[-1]])
            lnew = length(new)
            limit = bounds_clip(new, gcanny)
            if (lnew/2) < length(limit) <= lnew:
                x0, y0, x1, y1 = limit
            else:
                x0, y0, x1, y1 = new
            new = x0, y0, x1, y1, lnew, lines[i, 5]
            lines[i] = new
        return lines

    if hori is not None:
        hori = _shorten(hori)
    if vert is not None:
        inters = np.transpose(inters, axes=(1, 0, 2))
        vert = _shorten(vert)

    if algorithm.debug:
        canvas = draw.lines(gcanny, hori, vert, annotate_number=True)
        draw.save("fix_length_byinter", canvas)
    return hori, vert


def add_outer_diagonal(lines, ll, kind):
    log.debug("adding missing outer diagonal lines...")
    if ll < 5:
        log.warning("less than 5 lines passed to add_outer, returning...")
        return lines

    def _add_outer(lines, where):
        limit = gcanny.shape[kind-1]
        if where == 0:
            ref = 0
            other = 1
        elif where == -1:
            ref = limit
            other = -2

        line0 = lines[where]
        line1 = lines[other]

        space_old = min(abs(line0[kind] - ref), abs(line0[kind+2] - ref))
        if space_old <= 5:
            log.debug(f"space_old == 0, line0 (kind = {kind})")
            log.debug(f"({where=})")
            return lines

        x0, x1 = 2*line0[0] - line1[0], 2*line0[2] - line1[2]
        y0, y1 = 2*line0[1] - line1[1], 2*line0[3] - line1[3]
        new = np.array([x0, y0, x1, y1, 0, line0[5]], dtype='int32')
        new = bounds_clip(new, gcanny)
        new[4] = length(new)

        if new[4] < (line0[4]*0.7):
            log.debug(f"add_outer_diagonal({kind=}):")
            log.debug(f"line is shorter than next ({where=})")
            return lines

        if where == -1:
            lines = np.append(lines, [new], axis=0)
        else:
            lines = np.insert(lines, 0, [new], axis=0)
        return lines

    lines = _add_outer(lines, 0)
    lines = _add_outer(lines, -1)

    if algorithm.debug and ll != len(lines):
        canvas = draw.lines(gcanny, lines)
        draw.save("add_outer_diagonal", canvas)
    return lines, len(lines)


def add_outer_warped(lines, ll, kind):
    log.debug("adding missing outer warped lines...")
    if ll < 5:
        log.warning("less than 5 lines passed to add_outer, returning...")
        return lines

    def _add_outer(lines, where, med):
        limit = gcanny.shape[kind-1]
        if where == 0:
            ref = 0
            other = 1
        elif where == -1:
            ref = limit
            other = -2

        line0 = lines[where]
        line1 = lines[other]
        if where == 0:
            space = line0[kind] - ref
        elif where == -1:
            space = ref - line0[kind]

        if space <= 0:
            return lines

        new = np.copy(line0)
        dx = med + (med - segments_distance(line0, line1))
        if where == -1:
            new[kind] += dx
            new[kind+2] += dx
            lines = np.append(lines, [new], axis=0)
        else:
            new[kind] -= dx
            new[kind+2] -= dx
            lines = np.insert(lines, 0, [new], axis=0)
        return lines

    dists = calculate_distances_warped(lines, kind)
    med = np.median(dists)
    lines = _add_outer(lines, 0, med)
    lines = _add_outer(lines, -1, med)

    if algorithm.debug and ll != len(lines):
        canvas = draw.lines(gcanny, lines)
        draw.save("add_outer_warped", canvas)
    return lines, len(lines)


def remove_fake_outer(lines, ll, kind):

    def _remove_fake_outer(lines, where):
        if where == 0:
            other = 1
        elif where == -1:
            other = -2

        line0 = lines[where, :4]
        line1 = lines[other, :4]

        dkind = min(abs(line1[kind] - line0[kind]),
                    abs(line1[kind+2] - line0[kind+2]))
        dother = max(abs(line1[kind-1] - line0[kind-1]),
                     abs(line1[kind+1] - line0[kind+1]))

        if dkind < 15 and dother > 5:
            lines = np.delete(lines, where, axis=0)
        return lines

    lines = _remove_fake_outer(lines, 0)
    lines = _remove_fake_outer(lines, -1)

    if algorithm.debug:
        canvas = draw.lines(gcanny, lines)
        draw.save("remove_fake_outer", canvas)
    return lines, len(lines)


def extend_outer(lines, ll, kind):
    log.debug("extending outer lines...")
    if ll < 5:
        log.warning("less than 5 lines passed to extend_outers, returning...")
        return lines

    def _extend_outer(lines, where):
        limit = gcanny.shape[kind-1]
        if where == 0:
            ref = 0
        elif where == -1:
            ref = limit

        line0 = lines[where]
        new = np.copy(line0)

        if abs(line0[kind] - ref) <= 0 or abs(line0[kind+2] - ref) <= 0:
            return lines

        dmin = min(abs(new[kind]-ref), abs(new[kind+2]-ref))
        if where == -1:
            new[kind] += dmin
            new[kind+2] += dmin
        else:
            new[kind] -= dmin
            new[kind+2] -= dmin
        new = np.array([new], dtype='int32')
        lines[where] = new
        return lines

    lines = _extend_outer(lines, 0)
    lines = _extend_outer(lines, -1)

    if algorithm.debug:
        canvas = draw.lines(gcanny, lines)
        draw.save("extend_outer", canvas)
    return lines, len(lines)


def filter_misdirected2(hori, vert):
    log.debug("filtering lines by angle with next line")

    def _calculate_diffs(lines):
        diffs = np.zeros((lines.shape[0], 2), dtype='int32')
        i = 0
        diffs[i, 0] = abs(lines[i+0, 5] - lines[i+1, 5])
        diffs[i, 1] = diffs[i, 0]
        for i in range(1, len(lines) - 1):
            diffs[i, 0] = abs(lines[i+0, 5] - lines[i-1, 5])
            diffs[i, 1] = abs(lines[i+0, 5] - lines[i+1, 5])
        i += 1
        diffs[i, 0] = abs(lines[i+0, 5] - lines[i-1, 5])
        diffs[i, 1] = diffs[i, 0]
        return diffs

    def _remove_misdirected(lines, diffs):
        d0 = np.median(diffs[:, 0])
        d1 = np.median(diffs[:, 1])
        med = round((d0 + d1)/2)
        log.debug(f"median diff between lines: {med}")
        for i in range(0, len(lines)):
            if diffs[i, 0] > (med*3) < diffs[i, 1]:
                lines = np.delete(lines, i, axis=0)
                return lines
        return lines

    old_lv = old_lh = 0
    while old_lh != len(hori) or old_lv != len(vert):
        old_lh, old_lv = len(hori), len(vert)
        diffs_hori = _calculate_diffs(hori)
        diffs_vert = _calculate_diffs(vert)
        hori = _remove_misdirected(hori, diffs_hori)
        vert = _remove_misdirected(vert, diffs_vert)

    if algorithm.debug:
        canvas = draw.lines(gcanny, hori, vert)
        draw.save("filter_misdirected2", canvas)
    return hori, vert


def calculate_distances_warped(lines, kind):
    dists = np.zeros((lines.shape[0], 2), dtype='int32')
    dists[0, 0] = (lines[1, kind] - lines[0, kind]
                   + lines[1, kind+2] - lines[0, kind+2])
    dists[0, 1] = dists[0, 0]
    for i in range(1, len(lines) - 1):
        dists[i, 0] = (lines[i+0, kind] - lines[i-1, kind] +
                       lines[i+0, kind+2] - lines[i-1, kind+2])
        dists[i, 1] = (lines[i+1, kind] - lines[i+0, kind] +
                       lines[i+1, kind+2] - lines[i+0, kind+2])
    i += 1
    dists[i, 0] = (lines[i+0, kind] - lines[i-1, kind]
                   + lines[i+0, kind+2] - lines[i-1, kind+2])
    dists[i, 1] = dists[i, 0]
    return dists / 2


def calculate_distances_diagonal(lines, kind):
    dists = np.zeros((lines.shape[0], 2), dtype='int32')
    dists[0, 0] = segments_distance(lines[0], lines[1])
    dists[0, 1] = dists[0, 0]
    for i in range(1, len(lines) - 1):
        dists[i, 0] = segments_distance(lines[i], lines[i-1])
        dists[i, 1] = segments_distance(lines[i], lines[i+1])
    i += 1
    dists[i, 0] = segments_distance(lines[i], lines[i-1])
    dists[i, 1] = dists[i, 0]
    return dists


def remove_wrong(lines, ll, kind):
    log.debug("removing wrong middle lines...")
    if ll < 7:
        log.warning("less than 7 lines passed to remove_wrong, returning...")
        return lines, ll

    def _remove_wrong(lines, dists):
        med1 = abs(lines[0, kind] - lines[-1, kind])/8
        med2 = abs(lines[0, kind+2] - lines[-1, kind+2])/8
        med = (med1 + med2)/2
        med3 = gcanny.shape[1] / 8
        med = (med + med3)/2
        log.debug(f"median distance between lines: {med}")
        for i in range(0, len(lines)):
            if abs(dists[i, 0] - med) > 20 and abs(dists[i, 1] - med) > 20:
                lines = np.delete(lines, i, axis=0)
                return lines
        return lines

    dists = calculate_distances_warped(lines, kind)
    lines = _remove_wrong(lines, dists)
    dists = calculate_distances_warped(lines, kind)
    lines = _remove_wrong(lines, dists)

    if algorithm.debug and ll != len(lines):
        canvas = draw.lines(gcanny, lines)
        draw.save("remove_wrong", canvas)
    return lines, len(lines)


def add_middle_warped(lines, ll, kind):
    log.debug("adding missing middle lines...")
    tol = 1.4
    if ll < 5 or ll > 10:
        log.warning(f"{ll} lines passed to add_middle_warped, returning...")
        return lines, ll

    def _insert(lines, i, x, y):
        x0, x1 = x
        y0, y1 = y
        new = np.array([[x0, y0, x1, y1,
                         lines[i-1, 4], lines[i-1, 5]]], dtype='int32')
        lines = np.insert(lines, i+1, new, axis=0)
        return lines

    def _add_middle_warped(lines, med, sign):
        dnext0 = abs(lines[0, kind] - lines[1, kind])
        dnext1 = abs(lines[1, kind] - lines[2, kind])
        dnext2 = abs(lines[2, kind] - lines[3, kind])
        dnext3 = abs(lines[3, kind] - lines[4, kind])
        if dnext0 > (dnext1*tol) and dnext0 > (dnext2*tol):
            if dnext0 > (dnext3*tol):
                new = np.copy(lines[0])
                dx = med
                new[kind] += (dx*sign)
                new[kind+2] += (dx*sign)
                lines = np.insert(lines, 1, [new], axis=0)
                return lines
        for i in range(2, len(lines) - 2):
            dprev2 = abs(lines[i-1, kind] - lines[i-2, kind])
            dprev1 = abs(lines[i+0, kind] - lines[i-1, kind])
            dthis0 = abs(lines[i+0, kind] - lines[i+1, kind])
            dnext1 = abs(lines[i+1, kind] - lines[i+2, kind])
            if dthis0 > (dprev1*tol) and dthis0 > (dnext1*tol):
                if dthis0 > (dprev2*tol):
                    new = np.copy(lines[i])
                    dx = med
                    new[kind] += (dx*sign)
                    new[kind+2] += (dx*sign)
                    lines = np.insert(lines, i+1, [new], axis=0)
                    return lines
        return lines

    dists = calculate_distances_warped(lines, kind)
    med = np.median(dists)
    lines = _add_middle_warped(lines, med, +1)
    lines = np.flip(lines, axis=0)
    lines = _add_middle_warped(lines, med, -1)
    lines = np.flip(lines, axis=0)

    if algorithm.debug and ll != len(lines):
        canvas = draw.lines(gcanny, lines)
        draw.save("add_middle_warped", canvas)
    return lines, len(lines)


def remove_outer(lines, ll, kind):
    log.debug("removing extra outer lines...")
    tolerance = 3
    limit = gcanny.shape[kind-1]
    if ll < 7:
        log.warning("less than 7 lines passed to remove_outer, returning...")
        return lines, ll

    d00 = abs(lines[1, kind] - 0)
    d01 = abs(lines[1, kind+2] - 0)
    space0 = min(d00, d01)
    d10 = abs(lines[-2, kind] - limit)
    d11 = abs(lines[-2, kind+2] - limit)
    space1 = min(d10, d11)
    if ll <= 9:
        if space0 < tolerance:
            lines = lines[1:]
        if space1 < tolerance:
            lines = lines[:-1]
    else:  # always remove the outest line when ll > 9
        if space0 < space1:
            lines = lines[1:]
        else:
            lines = lines[:-1]

    if algorithm.debug and ll != len(lines):
        canvas = draw.lines(gcanny, lines)
        draw.save("remove_outer", canvas)
    return lines, len(lines)


def length_theta(hori, vert=None, abs_angle=False):
    if abs_angle:
        angle = theta_abs
    else:
        angle = theta

    def _calculate(lines):
        dummy = np.empty((lines.shape[0], 6), dtype='int32')
        dummy[:, 0:4] = lines[:, 0:4]
        lines = dummy[np.argsort(dummy[:, 0])]

        for i, line in enumerate(lines):
            x0, y0, x1, y1, r, t = line[:6]
            lines[i, 4] = length((x0, y0, x1, y1))
            lines[i, 5] = angle((x0, y0, x1, y1))*100
        return lines

    if vert is not None:
        vert = _calculate(vert)
    hori = _calculate(hori)
    return hori, vert


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


def split(lines):
    log.debug("spliting lines into vertical and horizontal...")

    def _check_split(hori, vert):
        if abs(np.median(hori[:, 5]) - np.median(vert[:, 5])) < 40*100:
            return False
        else:
            return True

    lines, _ = length_theta(lines)
    angles = lines[:, 5]
    try:
        limits = jenks_breaks(angles, n_classes=3)
    except Exception:
        return None, None, 0, 0

    a0 = angles[angles <= limits[1]]
    a1 = angles[(limits[1] < angles) & (angles <= limits[2])]
    a2 = angles[limits[2] < angles]
    centers = [np.median(a0), np.median(a1), np.median(a2)]

    d0 = abs(centers[0] - centers[1])
    d1 = abs(centers[0] - centers[2])
    d2 = abs(centers[1] - centers[2])

    maxdiff = 22.5 * 100
    dd0 = d0 < maxdiff and d1 > maxdiff and d2 > maxdiff
    dd1 = d1 < maxdiff and d0 > maxdiff and d2 > maxdiff
    dd2 = d2 < maxdiff and d0 > maxdiff and d1 > maxdiff

    if dd0 or dd1 or dd2:
        limits = jenks_breaks(angles, n_classes=2)
        hori = lines[angles <= limits[1]]
        vert = lines[limits[1] < angles]
        if not _check_split(hori, vert):
            return None, None, 0, 0
    else:
        for line in lines:
            if line[5] < (-45 * 100):
                line[5] = -line[5]
        angles = lines[:, 5]
        limits = jenks_breaks(angles, n_classes=2)
        hori = lines[angles <= limits[1]]
        vert = lines[limits[1] < angles]
        if not _check_split(hori, vert):
            return None, None, 0, 0

    if abs(np.median(vert[:, 5])) < abs(np.median(hori[:, 5])):
        aux = vert
        vert = hori
        hori = aux

    for line in vert:
        if line[1] > line[3]:
            a1, b1 = line[0], line[1]
            line[0], line[1] = line[2], line[3]
            line[2], line[3] = a1, b1
    return hori, vert, len(hori), len(vert)


def filter_misdirected(hori, vert):
    log.debug("filtering lines by angle accoring to direction...")
    tolerance = 15 * 100
    changed = False

    def _filter(lines):
        nonlocal changed
        angle = np.median(lines[:, 5])
        right = np.abs(lines[:, 5] - angle) <= tolerance
        changed = np.any(~right)
        return lines[right]

    hori = _filter(hori)
    vert = _filter(vert)

    if algorithm.debug and changed:
        canvas = draw.lines(gcanny, hori, vert)
        draw.save("filter_misdirected", canvas)
    return hori, vert


def filter_intersecting(hori, vert):
    log.debug("filtering lines by number of intersections ...")
    limit = True
    changed = False

    def _filter(lines):
        nonlocal changed
        for i, line in enumerate(lines):
            inters = intersect.calculate_all(np.array([line]), lines, limit)
            if inters is None or len(inters) == 0:
                continue
            elif inters.shape[1] >= 4:
                lines = np.delete(lines, i, axis=0)
                changed = True
                break
        return lines

    for j in range(3):
        hori = _filter(hori)
        vert = _filter(vert)

    if algorithm.debug and changed:
        canvas = draw.lines(gcanny, hori, vert)
        draw.save("filter_intersecting", canvas)
    return hori, vert


def filter_not_right(lines):
    lines, _ = length_theta(lines)
    remove = np.zeros(lines.shape[0], dtype='uint8')
    changed = False

    for i, t in enumerate(lines[:, 5]):
        if abs(t - 90*100) > 4*100 and abs(t + 90*100) > 4*100:
            if abs(t) > 4*100:
                remove[i] = 1
                changed = True
    lines = lines[remove == 0]

    if algorithm.debug and changed:
        canvas = draw.lines(gcanny, lines)
        draw.save("filter_not_right", canvas)
    return lines


def bounds_clip(line, canny):
    log.debug("shortening2...")
    limit = canny.shape[0]-1

    dx = line[2] - line[0]
    dy = line[3] - line[1]
    if dx != 0:
        a = dy/dx
    else:
        a = 0
    if dy != 0:
        b = dx/dy
    else:
        b = 0

    if line[0] < 0:
        line[1] = line[1] - a*(line[0] - 0)
        line[0] = 0
    if line[2] < 0:
        line[3] = line[3] - a*(line[2] - 0)
        line[2] = 0
    if line[1] < 0:
        line[0] = line[0] - b*(line[1] - 0)
        line[1] = 0
    if line[3] < 0:
        line[2] = line[2] - b*(line[3] - 0)
        line[3] = 0

    if line[0] > limit:
        line[1] = line[1] - a*(line[0] - limit)
        line[0] = limit
    if line[2] > limit:
        line[3] = line[3] - a*(line[2] - limit)
        line[2] = limit
    if line[1] > limit:
        line[0] = line[0] - b*(line[1] - limit)
        line[1] = limit
    if line[3] > limit:
        line[2] = line[2] - b*(line[3] - limit)
        line[3] = limit

    return line
