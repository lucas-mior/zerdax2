import numpy as np
import cv2
import logging as log

import algorithm as algo
import constants as consts
import drawings as draw


def warp(canny, corners):
    log.debug("transforming perspective...")
    top_left = corners[0]
    top_right = corners[1]
    bot_right = corners[2]
    bot_left = corners[3]
    orig_points = np.array((top_left, top_right,
                            bot_right, bot_left), dtype="float32")

    width = consts.warped_dimension - 1
    height = consts.warped_dimension - 1

    newshape = [[0, 0], [width, 0], [width, height], [0, height]]
    newshape = np.array(newshape, dtype='float32')
    warp_matrix = cv2.getPerspectiveTransform(orig_points, newshape)
    _, warp_inverse_matrix = cv2.invert(warp_matrix)

    canny_warped = cv2.warpPerspective(canny, warp_matrix, (width, height))
    if algo.debug:
        draw.save("canny_warped", canny_warped)

    return canny_warped, warp_inverse_matrix
