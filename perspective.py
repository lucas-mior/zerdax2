import numpy as np
import cv2
import logging as log

import constants as consts
import drawings as draw

WLEN = 512


def warp(canny, corners):
    log.debug("transforming perspective...")
    bot_right = corners[0]
    bot_left = corners[1]
    top_right = corners[2]
    top_left = corners[3]
    orig_points = np.array((top_left, top_right,
                            bot_right, bot_left), dtype="float32")

    width = WLEN
    height = WLEN

    newshape = np.array([[0, 0], [width-1, 0],
                        [width-1, height-1], [0, height-1]], dtype="float32")

    warp_matrix = cv2.getPerspectiveTransform(orig_points, newshape)
    _, warp_inverse_matrix = cv2.invert(warp_matrix)

    canny_warped = cv2.warpPerspective(canny, warp_matrix, (width, height))
    draw.save("canny_warped", canny_warped)

    return canny_warped, warp_inverse_matrix
