import numpy as np
import logging as log
from jenkspy import jenks_breaks

import algorithm as algo
import lines as li
import constants as consts
import intersect as intersect
import drawings as draw


def split(lines):
    log.info("spliting lines into vertical and horizontal...")

    def _check_split(vert, hori):
        if abs(np.median(hori[:, 5]) - np.median(vert[:, 5])) < 40*100:
            return False
        else:
            return True

    lines, _ = li.length_theta(lines)
    angles = lines[:, 5]

    limits = jenks_breaks(angles, n_classes=3)

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
        limits = jenks_breaks(angles, n_classes=2)
        hori = lines[angles <= limits[1]]
        vert = lines[limits[1] < angles]
        if not _check_split(vert, hori):
            return None, None
    else:
        for line in lines:
            if line[5] < (-45 * 100):
                line[5] = -line[5]
        angles = lines[:, 5]
        limits = jenks_breaks(angles, n_classes=2)
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


def filter_misdirected(vert, hori, canny_3channels):
    log.info("filtering lines by angle accoring to direction...")
    tolerance = consts.angle_tolerance

    def _filter(lines):
        angle = np.median(lines[:, 5])
        right = np.abs(lines[:, 5] - angle) <= tolerance
        return lines[right]

    vert = _filter(vert)
    hori = _filter(hori)

    if algo.debug:
        canvas = draw.lines(canny_3channels, vert, hori)
        draw.save("filter_misdirected", canvas)
    return vert, hori


def filter_intersecting(vert, hori=None):
    log.info("filtering lines by number of intersections ...")
    limit = True

    def _filter(lines):
        for i, line in enumerate(lines):
            inters = intersect.calculate_all(np.array([line]), lines, limit)
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


def filter_not_right(lines, canny_3channels):
    lines, _ = li.length_theta(lines)
    remove = np.zeros(lines.shape[0], dtype='uint8')

    for i, t in enumerate(lines[:, 5]):
        if abs(t - 90*100) > 4*100 and abs(t + 90*100) > 4*100:
            if abs(t) > 4*100:
                remove[i] = 1
    lines = lines[remove == 0]

    if algo.debug:
        canvas = draw.lines(canny_3channels, lines)
        draw.save("filter90", canvas)
    return lines
