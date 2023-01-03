import numpy as np


def bundle_lines(lines, min_dist=8, min_angle=15):
    groups = merge_lines_into_groups(lines, min_dist, min_angle)
    merged_lines = []
    for group in groups:
        merged_lines.append(merge_line_segments(group))

    return np.array(merged_lines)


def check_is_line_different(line1, groups, min_dist, min_angle):

    def _get_angle(line):
        dy = abs((line[3] - line[1]))
        dx = abs((line[2] - line[0]))
        orient = np.arctan2(dy, dx)
        return np.rad2deg(orient)

    def _get_dist(a_line, b_line):
        dist1 = min_distance(b_line[0:2], b_line[2:4], a_line[0:2])
        dist2 = min_distance(b_line[0:2], b_line[2:4], a_line[2:4])
        dist3 = min_distance(a_line[0:2], a_line[2:4], b_line[0:2])
        dist4 = min_distance(a_line[0:2], a_line[2:4], b_line[2:4])
        return min(dist1, dist2, dist3, dist4)

    for group in groups:
        for line2 in group:
            d = _get_dist(line2, line1)
            if d < min_dist:
                angle1 = _get_angle(line1)
                angle2 = _get_angle(line2)
                phi = abs(angle1 - angle2)
                if phi < min_angle or (d <= 1 and phi <= (min_angle+2)):
                    group.append(line1)
                    return False
    return True


def dist_point_to_line(point, line):
    px, py = point
    x1, y1, x2, y2 = line[0:4]

    def _line_mag(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        line_mag = np.sqrt(dx*dx + dy*dy)
        return line_mag

    lmag = _line_mag(x1, y1, x2, y2)
    if lmag < 0.00000001:
        dist_point_to_line = 9999
        return dist_point_to_line

    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (lmag * lmag)

    if (u < 0.00001) or (u > 1):
        # closest point does not fall within the line segment,
        # take the shorter distance to an endpoint
        ix = _line_mag(px, py, x1, y1)
        iy = _line_mag(px, py, x2, y2)
        if ix > iy:
            dist_point_to_line = iy
        else:
            dist_point_to_line = ix
    else:
        # Intersecting point is on the line, use the formula
        # ix = x1 + u * (x2 - x1)
        # iy = y1 + u * (y2 - y1)
        # dist_point_to_line = _line_mag(px, py, ix, iy)
        dist_point_to_line = 0

    return dist_point_to_line


def merge_lines_into_groups(lines, min_dist, min_angle):
    groups = []  # all lines groups are here
    # first line will create new group every time
    groups.append([lines[0]])
    # if line is different from existing gropus, create a new group
    for line_new in lines[1:]:
        if check_is_line_different(line_new, groups, min_dist, min_angle):
            groups.append([line_new])

    return groups


def merge_line_segments(lines):
    if len(lines) == 1:
        return np.block([lines[0][0:2], lines[0][2:4]])

    if len(lines) % 2 == 0:
        lines = sorted(lines, key=lambda x: x[4])
        lines = lines[1:]

    lines = np.array(lines)
    theta = np.median(lines[:, 5])
    P = lines[lines[:, 5] == theta]
    P = P[np.argmax(P[:, 4])]

    return np.block([P[0:2], P[2:4]])


def min_distance(A, B, E):
    # vector AB
    AB = [None, None]
    AB[0] = B[0] - A[0]
    AB[1] = B[1] - A[1]

    # vector BP
    BE = [None, None]
    BE[0] = E[0] - B[0]
    BE[1] = E[1] - B[1]

    # vector AP
    AE = [None, None]
    AE[0] = E[0] - A[0]
    AE[1] = E[1] - A[1]

    # Variables to store dot product

    # Calculating the dot product
    AB_BE = AB[0] * BE[0] + AB[1] * BE[1]
    AB_AE = AB[0] * AE[0] + AB[1] * AE[1]

    # Minimum distance from
    # point E to the line segment
    reqAns = 0

    # Case 1
    if (AB_BE > 0):
        # Finding the magnitude
        y = E[1] - B[1]
        x = E[0] - B[0]
        reqAns = np.sqrt(x*x + y*y)

    # Case 2
    elif (AB_AE < 0):
        y = E[1] - A[1]
        x = E[0] - A[0]
        reqAns = np.sqrt(x*x + y*y)

    # Case 3
    else:
        # Finding the perpendicular distance
        x1 = AB[0]
        y1 = AB[1]
        x2 = AE[0]
        y2 = AE[1]
        mod = np.sqrt(x1*x1 + y1*y1)
        reqAns = abs(x1*y2 - y1*x2) / mod

    return reqAns
