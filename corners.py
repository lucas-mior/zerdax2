import numpy as np
import logging as log

import lines as lines
import intersect as intersect


def find(canny):
    vert, hori = lines.find_baselines(canny)
    vert, hori = lines.fix_length_byinter(canny.shape, vert, hori)
    lv, lh = lines.check_save("fix_length_byinter0", vert, hori, -1, -1)

    if lv == 0 or lh == 0:
        return None

    vert, lv = lines.add_outer(vert, lv, 0, canny.shape)
    hori, lh = lines.add_outer(hori, lh, 1, canny.shape)
    inters = intersect.calculate_all(vert, hori)

    log.debug("calculating 4 corners of board...")
    inters = inters.reshape((-1, 2))
    psum = np.zeros((inters.shape[0], 3), dtype='int32')
    psub = np.zeros((inters.shape[0], 3), dtype='int32')

    psum[:, 0] = inters[:, 0]
    psum[:, 1] = inters[:, 1]
    psum[:, 2] = inters[:, 0] + inters[:, 1]
    psub[:, 0] = inters[:, 0]
    psub[:, 1] = inters[:, 1]
    psub[:, 2] = inters[:, 0] - inters[:, 1]

    top_left = psum[np.argmin(psum[:, 2])][0:2]
    top_right = psub[np.argmax(psub[:, 2])][0:2]
    bot_right = psum[np.argmax(psum[:, 2])][0:2]
    bot_left = psub[np.argmin(psub[:, 2])][0:2]
    return np.array([top_left, top_right, bot_right, bot_left], dtype='int32')
