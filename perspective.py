import numpy as np
import cv2
import logging as log

import constants as consts
import drawings as draw

minlen0 = consts.min_line_length
canny3ch = None
WLEN = 512


def warp(canny, corners):
    log.debug("transforming perspective...")
    BR = corners[0]
    BL = corners[1]
    TR = corners[2]
    TL = corners[3]
    orig_points = np.array(((TL[0], TL[1]), (TR[0], TR[1]),
                            (BR[0], BR[1]), (BL[0], BL[1])), dtype="float32")

    width = WLEN
    height = WLEN

    newshape = np.array([[0, 0], [width-1, 0],
                        [width-1, height-1], [0, height-1]], dtype="float32")

    warp_matrix = cv2.getPerspectiveTransform(orig_points, newshape)
    _, warp_inverse_matrix = cv2.invert(warp_matrix)

    canny_warped = cv2.warpPerspective(canny, warp_matrix, (width, height))
    draw.save("canny_warped", canny_warped)

    return canny_warped, warp_inverse_matrix
