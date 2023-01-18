import numpy as np
import logging as log
import constants as consts
import lines as li

min_dist = consts.min_dist_to_separate_lines
min_angle = consts.min_angle_to_separate_lines


def bundle_verthori(vert, hori):
    vert = bundle_lines(vert)
    hori = bundle_lines(hori)
    return vert, hori


def bundle_lines(lines, min_dist=min_dist, min_angle=min_angle):
    log.info("bundling similar lines together...")
    lines, _ = li.radius_theta(lines, abs_angle=False)
    groups = merge_lines_into_groups(lines, min_dist, min_angle)
    merged_lines = []
    for group in groups:
        line = merge_line_segments(group)
        merged_lines.append(line)

    return np.array(merged_lines, dtype='int32')


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


def merge_lines_into_groups(lines, min_dist, min_angle):
    log.debug("merging lines into groups...")
    groups = []
    groups.append([lines[0]])

    for line_new in lines[1:]:
        if check_is_line_different(line_new, groups, min_dist, min_angle):
            groups.append([line_new])

    return groups


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
    """ distance between two segments in the plane:
      one segment is (x11, y11) to (x12, y12)
      the other is   (x21, y21) to (x22, y22)
    """
    log.debug("calculating distance between line segments...")
    x11, y11, x12, y12 = line1[:4]
    x21, y21, x22, y22 = line2[:4]
    if segments_intersect(x11, y11, x12, y12, x21, y21, x22, y22):
        return 0
    # try each of the 4 vertices w/the other segment
    distances = []
    distances.append(point_segment_distance(x11, y11, x21, y21, x22, y22))
    distances.append(point_segment_distance(x12, y12, x21, y21, x22, y22))
    distances.append(point_segment_distance(x21, y21, x11, y11, x12, y12))
    distances.append(point_segment_distance(x22, y22, x11, y11, x12, y12))
    return min(distances)


def segments_intersect(x11, y11, x12, y12, x21, y21, x22, y22):
    log.debug("checking if segments intersect...")
    """ whether two segments in the plane intersect:
      one segment is (x11, y11) to (x12, y12)
      the other is   (x21, y21) to (x22, y22)
    """
    dx1 = x12 - x11
    dy1 = y12 - y11
    dx2 = x22 - x21
    dy2 = y22 - y21
    delta = dx2 * dy1 - dy2 * dx1
    if delta == 0:
        return False  # parallel segments

    s = (dx1*(y21 - y11) + dy1*(x11 - x21)) / delta
    t = (dx2*(y11 - y21) + dy2*(x21 - x11)) / (-delta)
    return (0 <= s <= 1) and (0 <= t <= 1)


def point_segment_distance(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == dy == 0:  # the segment's just a point
        dx = px - x1
        dy = py - y1
        return np.sqrt(dx*dx + dy*dy)

    # Calculate the t that minimizes the distance.
    t = ((px - x1)*dx + (py - y1)*dy) / (dx*dx + dy*dy)

    # See if this represents one of the segment's
    # end points or a point in the middle.
    if t < 0:
        dx = px - x1
        dy = py - y1
    elif t > 1:
        dx = px - x2
        dy = py - y2
    else:
        near_x = x1 + t*dx
        near_y = y1 + t*dy
        dx = px - near_x
        dy = py - near_y

    return np.sqrt(dx*dx + dy*dy)
