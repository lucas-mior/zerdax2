import numpy as np
import logging as log

import lines as lines
import intersections as intersections


def find(canny):
    log.info("finding all lines of board...")
    ww = canny.shape[1]
    hh = canny.shape[0]

    vert, hori = lines.find_baselines(canny)
    vert, hori = lines.fix_length_byinter(ww, hh, vert, hori)
    lv, lh = lines.check_save("fix_length_byinter0", vert, hori, -1, -1)

    if lv == 0 or lh == 0:
        return None, None

    vert, lv = lines.add_outer(vert, lv, 0, ww, hh)
    hori, lh = lines.add_outer(hori, lh, 1, ww, hh)
    inters = intersections.calculate_all(vert, hori)
    corners = calculate(inters)
    return corners


def calculate(inters):
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
