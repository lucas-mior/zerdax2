import numpy as np
import cv2
import logging as log
from jenkspy import jenks_breaks

import algorithm as algo
import intersections as intersections
import constants as consts
import drawings as draw
from c_load import segments_distance
from c_load import lines_bundle

minlen0 = consts.min_line_length
canny3ch = None
WLEN = 512


def find_wlines(canny):
    image_width = canny.shape[1]
    image_height = canny.shape[0]
    distv = round(image_width/23)
    disth = round(image_height/23)
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
        lines_hough = filter_90(lines_hough)
        canvas = draw.lines(canny3ch, lines_hough)
        draw.save("filter90", canvas)

        vert, hori = split(lines_hough)
        lv, lh = check_save("split", vert, hori, 0, 0)
        vert, hori = sort(vert, hori)
        lv, lh = check_save("sort", vert, hori, 0, 0)
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

    vert, hori = sort(vert, hori)
    lv, lh = check_save("sort", vert, hori, 0, 0)
    return vert, hori


def find_baselines(canny):
    image_width = canny.shape[1]
    image_height = canny.shape[0]
    distv = round(image_width/23)
    disth = round(image_height/23)
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
        vert, hori = split(lines_hough)
        lv, lh = check_save("split", vert, hori, 0, 0)
        vert, hori = filter_byangle(vert, hori)
        lv, lh = check_save("filter_byangle", vert, hori, lv, lh)
        vert, hori = sort(vert, hori)
        lv, lh = check_save("sort", vert, hori, 0, 0)
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
        canvas = draw.lines(canny3ch, vert, hori)
        draw.save(f"canny{lv=}_{lh=}", canvas)
    if lv < 6 or lh < 6:
        log.error("Less than 6 lines found in at least one direction")
        canvas = draw.lines(canny3ch, vert, hori)
        draw.save(f"canny{lv=}_{lh=}", canvas)
        return None, None

    vert, hori = sort(vert, hori)
    lv, lh = check_save("sort", vert, hori, 0, 0)
    return vert, hori


def split(lines):
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
            inters = intersections.calculate_all(np.array([line]), lines,
                                                 limit=True)
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


def filter_90(lines):
    remove = np.zeros(lines.shape[0], dtype='uint8')

    for i, t in enumerate(lines[:, 5]):
        if abs(t - 90*100) > 4*100 and abs(t + 90*100) > 4*100:
            if abs(t) > 4*100:
                remove[i] = 1

    return lines[remove == 0]


def sort(vert, hori=None, k=0):
    log.debug("sorting lines by position and respective direction...")

    def _create(lines, kind):
        positions = np.empty(lines.shape[0], dtype='int32')
        for i, line in enumerate(lines):
            inter = intersections.calculate_single(line, kind=kind)
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


def fix_length_byinter(image_width, image_height, vert, hori=None):
    inters = intersections.calculate_extern(vert, hori)
    if inters is None:
        log.debug("fix_length by inter did not find any intersection")
        return vert, hori

    def _shorten(lines):
        for i, inter in enumerate(inters):
            line = lines[i]
            a, b = inter[0], inter[-1]
            new = np.array([a[0], a[1], b[0], b[1]], dtype='int32')
            limit = limit_by_dimensions(new, image_width, image_height)
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
    annotate_number = True
    lv = 0 if vert is None else len(vert)
    lh = 0 if hori is None else len(hori)

    if old_lv == lv and old_lh == lh:
        return old_lv, old_lh

    if algo.debug:
        if lv > 12 or lh > 12:
            annotate_number = False
        canvas = draw.lines(canny3ch, vert, hori, annotate_number)
        draw.save(title, canvas)
    return lv, lh


def add_outer(lines, ll, k, image_width, image_height, force=False):
    log.info("adding missing outer lines...")
    tol = consts.outer_tolerance
    if force:
        tol = 1
    if ll < 5:
        log.warning("only 5 lines passed to add_outer. returning")
        return lines

    def _add_outer(lines, where):
        limit = image_width if k == 0 else image_height
        if where == 0:
            ref = 0
            other = 1
        elif where == -1:
            ref = limit
            other = -2

        line0 = lines[where]
        line1 = lines[other]

        x = (2*line0[0] - line1[0], 2*line0[2] - line1[2])
        y = (2*line0[1] - line1[1], 2*line0[3] - line1[3])
        if abs(line0[k] - ref) > tol and abs(line0[k+2] - ref) > tol:
            x0, x1 = x
            y0, y1 = y
            new = np.array([x0, y0, x1, y1], dtype='int32')
            inters = limit_by_dimensions(new, image_width, image_height)
            if len(inters) < 2:
                return lines
            elif len(inters) > 2:
                ls = np.array([[inters[0], inters[1]],
                               [inters[0], inters[2]],
                               [inters[1], inters[2]]])
                lens = [length(le.flatten()) for le in ls]
                inters = ls[np.argmax(lens)]
            if inters[0, k] <= limit and inters[1, k] <= limit:
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

    lines = _add_outer(lines, 0)
    ll, _ = check_save("add_outer0", lines, None, ll, 0)
    lines = _add_outer(lines, -1)
    ll, _ = check_save("add_outer1", lines, None, ll, 0)

    return lines, ll


def rem_middle(lines, ll):
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


def rem_wrong(lines, ll):
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


def add_middle(lines, ll):
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


def rem_outer(lines, ll, k, image_width, image_height, force=False):
    log.debug("removing extra outer lines...")
    tol = consts.outer_tolerance
    if k == 0:
        limit = image_width
    else:
        limit = image_height

    d00 = abs(lines[1, k] - 0)
    d01 = abs(lines[1, k+2] - 0)
    d0 = min(d00, d01)
    d10 = abs(lines[-2, k] - limit)
    d11 = abs(lines[-2, k+2] - limit)
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


def limit_by_dimensions(inters, image_width, image_height):
    i0 = intersections.calculate_single(inters, image_width, image_height, kind=0)
    i1 = intersections.calculate_single(inters, image_width, image_height, kind=1)
    i2 = intersections.calculate_single(inters, image_width, image_height, kind=2)
    i3 = intersections.calculate_single(inters, image_width, image_height, kind=3)
    inters = np.array([i0, i1, i2, i3], dtype='int32')

    inters = inters[(inters[:, 0] >= 0) & (inters[:, 1] >= 0) &
                    (inters[:, 0] <= image_width) & (inters[:, 1] <= image_height)]
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
