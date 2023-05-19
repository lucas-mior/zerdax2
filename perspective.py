import numpy as np
from numpy.linalg import det
import cv2
import logging as log
from jenkspy import jenks_breaks

import algorithm as algo
import constants as consts
import drawings as draw
from c_load import segments_distance
from c_load import lines_bundle

minlen0 = consts.min_line_length
canny3ch = None
WLEN = 512


def transform(canny, corners):
    print("transforming perspective...")
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
    _, warp_invmatrix = cv2.invert(warp_matrix)

    canny_warp = cv2.warpPerspective(canny, warp_matrix, (width, height))
    draw.save("cannywarp", canny_warp)

    return canny_warp, warp_invmatrix
