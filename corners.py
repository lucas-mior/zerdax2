import numpy as np
import logging as log

import lines as lines
import intersect as intersect
import constants as consts


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
    corners = broad([top_left, top_right, bot_right, bot_left], canny.shape)
    return corners


def broad(corners, image_shape):
    margin = consts.corners_margin
    width = image_shape[1] - 1
    height = image_shape[0] - 1
    top_left = corners[0]
    top_right = corners[1]
    bot_right = corners[2]
    bot_left = corners[3]
    top_left[0] = max(0,       top_left[0] - margin)
    top_left[1] = max(0,       top_left[1] - margin)
    top_right[0] = min(width,  top_right[0] + margin)
    top_right[1] = max(0,      top_right[1] - margin)
    bot_right[0] = min(width,  bot_right[0] + margin)
    bot_right[1] = min(height, bot_right[1] + margin)
    bot_left[0] = max(0,       bot_left[0] - margin)
    bot_left[1] = min(height,  bot_left[1] + margin)
    return np.array([top_left, top_right, bot_right, bot_left], dtype='int32')
