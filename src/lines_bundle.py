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
    for group in groups:
        for line2 in group:
            dtheta = abs(line1[5] - line2[5])
            if dtheta < min_angle:
                dist = segments_distance(line2, line1)
                if dist < min_dist:
                    group.append(line1)
                    return False
            elif (dtheta <= (min_angle+2)):
                dist = segments_distance(line2, line1)
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


def segments_distance(line1, line2):
    log.debug("calculating distance between line segments...")
    if segments_intersect(line1[:4], line2[:4]):
        return 0
    # try each of the 4 vertices w/the other segment
    distances = []
    distances.append(point_seg_distance(line1[0:2], line2[:4]))
    distances.append(point_seg_distance(line1[2:4], line2[:4]))
    distances.append(point_seg_distance(line2[0:2], line1[:4]))
    distances.append(point_seg_distance(line2[2:4], line1[:4]))
    return min(distances)


def segments_intersect(line1, line2):
    log.debug("checking if segments intersect...")
    x0, y0, x1, y1 = line1[:4]
    xx0, yy0, xx1, yy1 = line2[:4]
    dx0 = x1 - x0
    dy0 = y1 - y0
    dx1 = xx1 - xx0
    dy1 = yy1 - yy0
    delta = dx1*dy0 - dy1*dx0
    if delta == 0:
        return False  # parallel segments

    s = (dx0*(yy0 - y0) + dy0*(x0 - xx0)) / delta
    t = (dx1*(y0 - yy0) + dy1*(xx0 - x0)) / (-delta)
    return (0 <= s <= 1) and (0 <= t <= 1)


def point_seg_distance(point, line):
    px, py = point
    x0, y0, x1, y1 = line
    dx = x1 - x0
    dy = y1 - y0
    if dx == dy == 0:  # the segment's just a point
        dx = px - x0
        dy = py - y0
        return np.sqrt(dx*dx + dy*dy)

    # Calculate the t that minimizes the distance.
    t = ((px - x0)*dx + (py - y0)*dy) / (dx*dx + dy*dy)

    # See if this represents one of the segment's
    # end points or a point in the middle.
    if t < 0:
        dx = px - x0
        dy = py - y0
    elif t > 1:
        dx = px - x1
        dy = py - y1
    else:
        near_x = x0 + t*dx
        near_y = y0 + t*dy
        dx = px - near_x
        dy = py - near_y

    return np.sqrt(dx*dx + dy*dy)
