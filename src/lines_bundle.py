import numpy as np
import logging as log

import constants as consts
import lines as li

min_dist = consts.min_dist_to_separate_lines
min_angle = consts.min_angle_to_separate_lines


def lines_bundle(lines, min_dist=min_dist, min_angle=min_angle):
    log.info("bundling similar lines together...")
    lines, _ = li.length_theta(lines, abs_angle=False)
    groups = merge_lines_into_groups(lines, min_dist, min_angle)
    merged_lines = []
    for group in groups:
        line = merge_line_segments(group)
        merged_lines.append(line)

    return np.array(merged_lines, dtype='int32')


def merge_lines_into_groups(lines, min_dist, min_angle):
    log.debug("merging lines into groups...")
    groups = []
    groups.append([lines[0]])

    for line_new in lines[1:]:
        if check_is_line_different(line_new, groups, min_dist, min_angle):
            groups.append([line_new])

    return groups


def check_is_line_different(line1, groups, min_dist, min_angle):
    log.debug("checking if line is different from lines in groups...")
    min_angle2 = min_angle + 2
    for group in groups:
        for line0 in group:
            dtheta = abs(line1[5] - line0[5])
            if dtheta < min_angle:
                dist = li.segments_distance(line0, line1)
                if dist < min_dist:
                    group.append(line1)
                    return False
            elif (dtheta <= min_angle2):
                dist = li.segments_distance(line0, line1)
                if dist <= 1:
                    group.append(line1)
                    return False
    return True


def merge_line_segments(lines):
    log.debug("merging line segments...")
    ll = len(lines)
    if ll == 1:
        return np.block([lines[0][0:2], lines[0][2:4]])

    if ll % 2 == 0:
        lines = sorted(lines, key=lambda x: x[4])
        lines = lines[1:]

    lines = np.array(lines)
    theta = np.median(lines[:, 5])
    P = lines[lines[:, 5] == theta]
    P = P[np.argmax(P[:, 4])]

    return np.block([P[0:2], P[2:4]])
