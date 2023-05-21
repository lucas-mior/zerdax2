import sys
import cv2
import numpy as np
import logging as log
from types import SimpleNamespace

import algorithm as algo
import squares as squares
import lines as lines
import intersect as intersect
import yolo_wrap as yolo
import fen as fen
import constants as consts
import drawings as draw
from c_load import lfilter

img = None
debug = False


def algorithm(filename):
    global img, debug
    debug = log.root.level < 20
    img = SimpleNamespace(filename=filename)
    img.basename = filename.rsplit(".", 1)[0]
    img.basename = img.basename.rsplit("/", 1)[-1]
    bad_picture_msg = f"{img.filename}: bad picture, change the camera angle"

    img.BGR = cv2.imread(img.filename)

    board = SimpleNamespace()
    board.box, board.pieces = yolo.detect_objects(img.filename)
    board.pieces = yolo.determine_colors(board.pieces, img.BGR)
    board.pieces = yolo.process_pieces(board.pieces)

    if (failed := board.box is None) or algo.debug:
        canvas = draw.boxes(img.BGR, board.pieces)
        draw.save("yolo", canvas)
        if failed:
            log.error(bad_picture_msg)
            return bad_picture_msg

    log.info(f"board detected: {board.box}")
    board.image, translate_params = crop_image(img.BGR, board.box)
    if board.image.shape[0] < consts.min_boardbox_height:
        log.error(bad_picture_msg)
        return bad_picture_msg

    canny = create_canny(board.image)

    board.corners = find_corners(canny)
    if board.corners is None:
        log.error(bad_picture_msg)
        return bad_picture_msg

    log.info(f"corners found: {board.corners}")
    canny_warped, warp_inverse_matrix = warp(canny, board.corners)

    vert, hori = lines.find_warped_lines(canny_warped)
    if vert is None or hori is None:
        log.error(bad_picture_msg)
        return bad_picture_msg

    inters = intersect.calculate_all(vert, hori)
    if (failed := inters.shape != (9, 9, 2)) or algo.debug:
        canvas = draw.points(canny_warped, inters)
        draw.save("intersections", canvas)
        if failed:
            log.error("there should be 81 intersections",
                      "in 9 rows and 9 columns")
            log.error(f"{inters.shape=}")
            log.error(bad_picture_msg)
            return bad_picture_msg

    inters = translate_inters(inters, warp_inverse_matrix, translate_params)
    if algo.debug:
        canvas = draw.points(img.BGR, inters)
        draw.save("translated_intersections", canvas)

    board.squares = squares.calculate(inters)
    if algo.debug:
        canvas = draw.squares(img.BGR, board.squares)
        draw.save("A1E4C5H8", canvas)

    board.squares, pieces = squares.fill(board.squares, board.pieces)
    board.squares, changed = squares.check_colors(img.BGR, board.squares)
    if algo.debug and changed:
        canvas = draw.squares(img.BGR, board.squares)
        draw.save("A1E4C5H8", canvas)

    board.fen = fen.generate(board.squares)
    fen.dump(board.fen)
    return board.fen


def crop_image(image, boardbox):
    log.info("cropping image to board box...")
    translate_params = SimpleNamespace()
    x0, y0, x1, y1 = boardbox
    margin = consts.margin
    x0, y0 = x0 - margin, y0 - margin
    x1, y1 = x1 + margin, y1 + margin
    cropped = image[y0:y1, x0:x1]

    log.info("reducing cropped image to default size...")
    width_board = consts.width_board
    resize_factor = width_board / cropped.shape[1]
    height_board = round(resize_factor * cropped.shape[0])
    cropped = cv2.resize(cropped, (width_board, height_board))

    translate_params.x0 = x0
    translate_params.y0 = y0
    translate_params.resize_factor = resize_factor

    if algo.debug:
        draw.save("cropped", cropped)
    return cropped, translate_params


def create_canny(image):
    log.info("pre-processing image...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsvalue = hsv[:, :, 2]

    log.info("applying distributed histogram equalization to image...")
    grid = (consts.tile_grid_size, consts.tile_grid_size)
    clip_limit = consts.clip_limit
    clahe = cv2.createCLAHE(clip_limit, grid)
    gray = clahe.apply(gray)
    hsvalue = clahe.apply(hsvalue)

    if algo.debug:
        draw.save("clahe_gray", gray)
        draw.save("clahe_hsvalue", hsvalue)

    log.info("finding edges for gray and hsvalue images...")
    canny_gray = find_edges(gray)
    canny_hsvalue = find_edges(hsvalue)
    canny = cv2.bitwise_or(canny_gray, canny_hsvalue)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    canny = cv2.morphologyEx(canny, cv2.MORPH_DILATE, kernel)
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

    if algo.debug:
        draw.save("canny_gray", canny_gray)
        draw.save("canny_hsvalue", canny_hsvalue)
        draw.save("canny", canny)
    return canny


def find_edges(image):
    log.info("filtering image...")
    f = np.array(image, dtype='float64')
    W = np.zeros(image.shape, dtype='float64')
    N = np.zeros(image.shape, dtype='float64')
    g = np.zeros(image.shape, dtype='float64')

    lfilter(f, f.shape[0], f.shape[1], W, N, g, 1)
    lfilter(g, f.shape[0], f.shape[1], W, N, f, 1)
    lfilter(f, f.shape[0], f.shape[1], W, N, g, 1)

    g = np.round(g)
    g = np.clip(g, 0, 255)
    g = np.array(g, dtype='uint8')
    if algo.debug:
        draw.save("lowpass", g)

    canny_mean_threshold = consts.canny_mean_threshold_filter
    threshold_high0 = consts.threshold_highfilter
    canny = find_canny(g, canny_mean_threshold, threshold_high0)
    return canny


def find_canny(image, canny_mean_threshold=8, threshold_high0=250):
    log.info(f"finding edges with Canny until "
             f"mean >= {canny_mean_threshold:0=.1f}...")

    got_canny = False

    canny_threshold_high_min = consts.canny_threshold_high_min
    canny_threshold_low_min = consts.canny_threshold_low_min

    threshold_high = threshold_high0
    while threshold_high >= canny_threshold_high_min:
        threshold_low = max(canny_threshold_low_min, round(threshold_high*0.8))
        while threshold_low >= canny_threshold_low_min:
            canny = cv2.Canny(image, threshold_low, threshold_high)
            mean = np.mean(canny)
            if mean >= canny_mean_threshold:
                log.info(f"{mean:0=.2f} >= {canny_mean_threshold:0=.1f},"
                         f" @ {threshold_low}, {threshold_high}")
                got_canny = True
                break
            else:
                log.debug(f"{mean:0=.2f} < {canny_mean_threshold:0=.1f},"
                          f" @ {threshold_low}, {threshold_high}")
                gain = canny_mean_threshold - mean
                diff = round(max(8, gain*8))
                if threshold_low <= canny_threshold_low_min:
                    break
                threshold_low = max(canny_mean_threshold, threshold_low - diff)

        if got_canny or (threshold_high <= canny_threshold_high_min):
            break

        diff = round(max(6, gain*(threshold_high/18)))
        threshold_high = max(canny_threshold_high_min, threshold_high - diff)

    if not got_canny:
        log.info(f"Failed to find edges with"
                 f"mean >= {canny_mean_threshold:0=.1f}")
        log.info(f"Last canny thresholds: {threshold_low, threshold_high}")
    return canny


def warp(canny, corners):
    log.debug("transforming perspective...")
    TL = corners[0]
    TR = corners[1]
    BR = corners[2]
    BL = corners[3]
    orig_points = np.array((TL, TR, BR, BL), dtype="float32")

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


def translate_inters(inters, warp_inverse_matrix, translate_params):
    # go back to original perspective
    inters = np.array(inters, dtype='float64')
    inters = cv2.perspectiveTransform(inters, warp_inverse_matrix)
    # scale to input size
    inters[:, :, 0] /= translate_params.resize_factor
    inters[:, :, 1] /= translate_params.resize_factor
    # position board bounding box
    inters[:, :, 0] += translate_params.x0
    inters[:, :, 1] += translate_params.y0
    inters = np.array(np.round(inters), dtype='int32')

    return inters


def find_corners(canny):
    vert, hori = lines.find_diagonal_lines(canny)
    if vert is None or hori is None:
        return None

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

    TL = psum[np.argmin(psum[:, 2])][0:2]
    TR = psub[np.argmax(psub[:, 2])][0:2]
    BR = psum[np.argmax(psum[:, 2])][0:2]
    BL = psub[np.argmin(psub[:, 2])][0:2]

    log.debug("broading 4 corners of board...")
    margin = consts.corners_margin

    width = canny.shape[1] - 1
    height = canny.shape[0] - 1

    TL[0] = max(0,      TL[0] - margin)
    TL[1] = max(0,      TL[1] - margin)
    TR[0] = min(width,  TR[0] + margin)
    TR[1] = max(0,      TR[1] - margin)
    BR[0] = min(width,  BR[0] + margin)
    BR[1] = min(height, BR[1] + margin)
    BL[0] = max(0,      BL[0] - margin)
    BL[1] = min(height, BL[1] + margin)

    corners = np.array([TL, TR, BR, BL], dtype='int32')
    if algo.debug:
        canvas = draw.corners(canny, corners)
        draw.save("corners", canvas)
    return corners


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        print(algorithm(filename))
