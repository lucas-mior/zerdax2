import numpy as np
import cv2
import logging as log

import algorithm as algo
import intersect as intersect
import constants as consts
import angles as angles
import drawings as draw
from c_load import segments_distance
from c_load import lines_bundle

hough_min_length0 = consts.min_line_length


def find_warped_lines(canny):
    log.debug("finding all lines of warped board...")
    image_shape = canny.shape
    distv = round(image_shape[1]/23)
    disth = round(image_shape[0]/23)
    min_lines_before_split = consts.min_lines_before_split

    angle = consts.hough_angle_resolution
    hough_min_length = hough_min_length0
    hough_max_gap = 4
    hough_threshold = round(hough_min_length0*1.1)
    ll = lv = lh = 0
    hori = vert = None
    while lv < 8 or lh < 8:
        if hough_threshold <= round(hough_min_length0/1.5):
            if (hough_min_length <= round(hough_min_length0/1.1)):
                break
        hough_min_length = max(hough_min_length - 5,
                               round(hough_min_length0 / 1.1))
        hough_max_gap = min(hough_max_gap + 2, round(hough_min_length0 / 4))
        hough_threshold = max(hough_threshold - 5,
                              round(hough_min_length0 / 1.5))
        lines, ll = hough_wrapper(canny, hough_threshold,
                                  hough_min_length, hough_max_gap)
        if ll < min_lines_before_split:
            hough_max_gap += 2
            continue

        lines = angles.filter_not_right(lines, canny)
        vert, hori = angles.split(lines)
        if vert is None or hori is None:
            continue

        vert, hori = sort(vert, hori, canny)
        vert, hori = bundle_lines(vert, distv, hori, disth, canny)

        lv, lh = len(vert), len(hori)
        ll = lv + lh
        log.info(f"{ll} # {lv},{lh} @ {angle}º, {hough_threshold=}, "
                 f"{hough_min_length=},{hough_max_gap=}")

    if (failed := lv < 6 or lh < 6) or algo.debug:
        canvas = draw.lines(canny, vert, hori)
        draw.save(f"canny{lv=}_{lh=}", canvas)
        if failed:
            log.error("Less than 6 lines found in at least one direction")
            return None, None

    vert, hori = sort(vert, hori, canny)
    vert, hori = fix_warped_lines(vert, hori, canny)
    return vert, hori


def find_baselines(canny):
    log.debug("finding all lines of board...")
    image_shape = canny.shape
    distv = round(image_shape[0]/23)
    disth = round(image_shape[1]/23)
    min_lines_before_split = consts.min_lines_before_split

    angle = consts.hough_angle_resolution
    hough_min_length = hough_min_length0
    hough_max_gap = 4
    hough_threshold = round(hough_min_length0*1.1)
    ll = lv = lh = 0
    hori = vert = None
    while lv < 8 or lh < 8:
        if hough_threshold <= round(hough_min_length0/1.5):
            if hough_min_length <= round(hough_min_length0/1.1):
                break
        hough_min_length = max(hough_min_length - 5,
                               round(hough_min_length0 / 1.1))
        hough_max_gap = min(hough_max_gap + 2, round(hough_min_length0 / 4))
        hough_threshold = max(hough_threshold - 5,
                              round(hough_min_length0 / 1.5))
        lines, ll = hough_wrapper(canny, hough_threshold,
                                  hough_min_length, hough_max_gap)
        if ll < min_lines_before_split:
            hough_max_gap += 2
            continue

        vert, hori = angles.split(lines)
        if vert is None or hori is None:
            continue

        vert, hori = angles.filter_misdirected(vert, hori, canny)
        vert, hori = sort(vert, hori, canny)
        vert, hori = bundle_lines(vert, distv, hori, disth, canny)
        vert, hori = angles.filter_intersecting(vert, hori, canny)

        lv, lh = len(vert), len(hori)
        ll = lv + lh
        log.info(f"{ll} # {lv},{lh} @ {angle}º, {hough_threshold=}, "
                 f"{hough_min_length=},{hough_max_gap=}")

    if (failed := lv < 6 or lh < 6) or algo.debug:
        canvas = draw.lines(canny, vert, hori)
        draw.save(f"canny{lv=}_{lh=}", canvas)
        if failed:
            log.error("Less than 6 lines found in at least one direction")
            return None, None

    vert, hori = sort(vert, hori, canny)
    return vert, hori


def hough_wrapper(canny, hough_threshold, hough_min_length, hough_max_gap):
    angle = consts.hough_angle_resolution
    hough_angle = np.deg2rad(angle)
    lines = cv2.HoughLinesP(canny, 1, hough_angle, hough_threshold,
                            None, hough_min_length, hough_max_gap)
    if lines is None:
        ll = 0
    else:
        lines = lines[:, 0, :]
        ll = len(lines)

    log.debug(f"{ll} @ {angle}, {hough_threshold=}, "
              f"{hough_min_length=}, {hough_max_gap=}")
    if algo.debug:
        canvas = draw.lines(canny, lines)
        draw.save("hough_lines", canvas)
    return lines, ll


def bundle_lines(vert, distv, hori, disth, canny):

    def _bundle_lines(lines, dist):
        bundled = np.zeros(lines.shape, dtype='int32')
        nlines = lines_bundle(lines, bundled, len(lines), dist)
        lines = bundled[:nlines]
        return lines

    hori = _bundle_lines(hori, disth)
    vert = _bundle_lines(vert, distv)

    if algo.debug:
        canvas = draw.lines(canny, vert, hori)
        draw.save("bundle_lines", canvas)
    return vert, hori


def fix_warped_lines(vert, hori, canny):

    def _fix_warped_lines(lines, kind):
        lines, ll = rem_wrong(lines, len(lines))
        lines, ll = add_outer(lines, ll, kind, canny.shape)
        lines, ll = rem_outer(lines, ll, kind, canny.shape)
        lines, ll = add_outer(lines, ll, kind, canny.shape, force=(ll < 9))
        lines, ll = add_middle(lines, ll)
        return lines, ll

    vert, lv = _fix_warped_lines(vert, 0)
    hori, lh = _fix_warped_lines(hori, 1)
    if lv != 9:
        vert, lv = _fix_warped_lines(vert, 0)
    if lh != 9:
        hori, lh = _fix_warped_lines(hori, 1)

    if lv != 9 or lh != 9:
        canvas = draw.lines(canny, vert, hori)
        draw.save(f"canny{lv=}_{lh=}", canvas)
        return None, None
    return vert, hori


def sort(vert, hori, canny):
    log.debug("sorting lines by position and respective direction...")

    def _create(lines, kind):
        positions = np.empty(lines.shape[0], dtype='int32')
        for i, line in enumerate(lines):
            inter = intersect.calculate_single(line, kind=kind)
            positions[i] = inter[kind]
        return positions

    positions = _create(hori, 1)
    hori = hori[np.argsort(positions)]
    positions = _create(vert, 0)
    vert = vert[np.argsort(positions)]

    if algo.debug:
        canvas = draw.lines(canny, vert, hori)
        draw.save("sort_lines", canvas)
    return vert, hori


def fix_length_byinter(image_shape, vert, hori=None):
    inters = intersect.calculate_extern(vert, hori)
    if inters is None:
        log.debug("fix_length by inter did not find any intersection")
        return vert, hori

    def _shorten(lines):
        for i, inter in enumerate(inters):
            line = lines[i]
            a, b = inter[0], inter[-1]
            new = np.array([a[0], a[1], b[0], b[1]], dtype='int32')
            limit = intersect.shorten(new, image_shape)
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


def add_outer(lines, ll, kind, image_shape, force=False):
    log.info("adding missing outer lines...")
    tol = consts.outer_tolerance
    if force:
        tol = 1
    if ll < 5:
        log.warning("Less than 5 lines passed to add_outer, returning...")
        return lines

    def _add_outer(lines, where):
        limit = image_shape[kind-1]
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
        if abs(line0[kind] - ref) > tol and abs(line0[kind+2] - ref) > tol:
            x0, x1 = x
            y0, y1 = y
            new = np.array([x0, y0, x1, y1], dtype='int32')
            inters = intersect.shorten(new, image_shape)
            if len(inters) < 2:
                return lines
            elif len(inters) > 2:
                segments = np.array([[inters[0], inters[1]],
                                     [inters[0], inters[2]],
                                     [inters[1], inters[2]]])
                lengths = [length(segment.flatten()) for segment in segments]
                inters = segments[np.argmax(lengths)]
            if inters[0, kind] <= limit and inters[1, kind] <= limit:
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
    lines = _add_outer(lines, -1)
    return lines, len(lines)


def rem_middle(lines, ll):
    log.info("reming missing middle lines...")
    tol = consts.middle_tolerance
    if ll < 7:
        log.warning("Less than 7 lines passed to rem_middle, returning...")
        return lines, ll

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
    return lines, ll


def rem_wrong(lines, ll):
    log.info("removing wrong middle lines...")
    tol = consts.middle_tolerance
    if ll < 7:
        log.warning("Less than 7 lines passed to rem_wrong, returning...")
        return lines, ll

    def _calc_distances(lines):
        dists = np.zeros((lines.shape[0], 2), dtype='int32')
        i = 0
        dists[i, 0] = segments_distance(lines[i+0], lines[i+1])
        dists[i, 1] = dists[i, 0]
        for i in range(1, len(lines) - 1):
            dists[i, 0] = segments_distance(lines[i+0], lines[i-1])
            dists[i, 1] = segments_distance(lines[i+0], lines[i+1])
        i += 1
        dists[i, 0] = segments_distance(lines[i+0], lines[i-1])
        dists[i, 1] = dists[i, 0]
        return dists

    def _rem_wrong(lines, dists):
        d0 = np.median(dists[:, 0])
        d1 = np.median(dists[:, 1])
        med = round((d0 + d1)/2)
        log.debug(f"median distance between lines: {med}")
        for i in range(0, len(lines)):
            if dists[i, 1] < (med/tol) and (med*tol) < dists[i, 0]:
                lines = np.delete(lines, i, axis=0)
                return lines
        return lines

    dists = _calc_distances(lines)
    lines = _rem_wrong(lines, dists)
    lines = _rem_wrong(lines, dists)
    return lines, len(lines)


def add_middle(lines, ll):
    log.info("adding missing middle lines...")
    tol = consts.middle_tolerance
    if ll < 5 or ll > 10:
        log.warning(f"{ll} lines passed to add_middle, returning...")
        return lines, ll

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
        if dnext0 > (dnext1*tol) and dnext0 > (dnext2*tol):
            if dnext0 > (dnext3*tol):
                x = (2*lines[1, 0] - lines[2, 0],
                     2*lines[1, 2] - lines[2, 2])
                y = (2*lines[1, 1] - lines[2, 1],
                     2*lines[1, 3] - lines[2, 3])
                lines = _insert(lines, 0, x, y)
                return lines
        for i in range(2, len(lines) - 3):
            dprev2 = segments_distance(lines[i-1], lines[i-2])
            dprev1 = segments_distance(lines[i+0], lines[i-1])
            dthis0 = segments_distance(lines[i+0], lines[i+1])
            dnext1 = segments_distance(lines[i+1], lines[i+2])
            dnext2 = segments_distance(lines[i+2], lines[i+3])
            if dthis0 > (dprev1*tol) and dthis0 > (dnext1*tol):
                if dthis0 > (dprev2*tol) and dthis0 > (dnext2*tol):
                    x = (2*lines[i, 0] - lines[i-1, 0],
                         2*lines[i, 2] - lines[i-1, 2])
                    y = (2*lines[i, 1] - lines[i-1, 1],
                         2*lines[i, 3] - lines[i-1, 3])
                    lines = _insert(lines, i, x, y)
                    return lines
        for i in range(1, len(lines) - 4):
            dprev0 = segments_distance(lines[i+0], lines[i-1])
            dnext0 = segments_distance(lines[i+0], lines[i+1])
            dnext1 = segments_distance(lines[i+1], lines[i+2])
            dnext2 = segments_distance(lines[i+2], lines[i+3])
            dnext3 = segments_distance(lines[i+3], lines[i+4])
            if dnext0 > (dnext1*tol) and dnext0 > (dnext2*tol):
                if dnext0 > (dprev0*tol) or dnext0 > (dnext3*tol):
                    x = (round((lines[i, 0] + lines[i+1, 0])/2),
                         round((lines[i, 2] + lines[i+1, 2])/2))
                    y = (round((lines[i, 1] + lines[i+1, 1])/2),
                         round((lines[i, 3] + lines[i+1, 3])/2))
                    lines = _insert(lines, i, x, y)
                    return lines
        return lines

    lines = _add_middle(lines)
    lines = np.flip(lines, axis=0)
    lines = _add_middle(lines)
    lines = np.flip(lines, axis=0)
    return lines, len(lines)


def rem_outer(lines, ll, kind, image_shape):
    log.debug("removing extra outer lines...")
    tolerance = consts.outer_tolerance
    limit = image_shape[kind-1]
    if ll < 7:
        log.warning("Less than 7 lines passed to rem_outer, returning...")
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

    return lines, len(lines)


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
