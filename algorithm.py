import sys
import cv2
import numpy as np
import logging as log
from types import SimpleNamespace

import squares
import lines
import intersect
import objects
import fen
import draw
from c_load import lfilter
import shutil

basename = ""
debug = False
WIDTH_INPUT = 1920
WIDTH_BOARD = 512  # used for board crop and perspective transform


def main(filename):
    global basename, debug
    debug = log.root.level < 20
    basename = str.rsplit(filename, ".", 1)[0]
    basename = str.rsplit(basename, "/", 1)[-1]
    bad_picture_msg = f"{filename}: bad picture, change the camera angle"

    BGR = cv2.imread(filename)

    aspect_ratio = WIDTH_INPUT / BGR.shape[1]
    height_input = round(BGR.shape[0] * aspect_ratio)
    BGR = cv2.resize(BGR, (WIDTH_INPUT, height_input))

    board = SimpleNamespace()
    board.box, board.pieces = objects.detect(BGR)

    if (failed := board.box is None) or debug:
        canvas = draw.boxes(BGR, board.pieces, board.box)
        draw.save("detection", canvas)
        if failed:
            log.error("could not find board on picture")
            log.error(bad_picture_msg)
            return bad_picture_msg

    log.info(f"board detected: {board.box}")
    board.image, translate_params = crop_image(BGR, board.box)
    if (h := board.image.shape[0]) < 280:
        log.error(f"too low perspective: {h}")
        log.error(bad_picture_msg)
        return bad_picture_msg
    elif h > (w := board.image.shape[1]):
        log.error(f"height ({h}) is bigger than width ({w}).")
        log.error("wrong board detected.")
        log.error(bad_picture_msg)
        return bad_picture_msg

    board.pieces = objects.remove_captured_pieces(board.pieces, board.box)
    pieces_image, pieces_params = crop_pieces(BGR, board.box, board.pieces)
    board.pieces[:, 0] -= pieces_params.x0
    board.pieces[:, 2] -= pieces_params.x0
    board.pieces[:, 1] -= pieces_params.y0
    board.pieces[:, 3] -= pieces_params.y0
    for piece in board.pieces:
        piece[0] *= pieces_params.resize_factor
        piece[1] *= pieces_params.resize_factor
        piece[2] *= pieces_params.resize_factor
        piece[3] *= pieces_params.resize_factor

    canvas = draw.boxes(pieces_image, board.pieces)
    draw.save("pieces_image", canvas)

    board.pieces = objects.determine_colors(board.pieces, pieces_image)
    canvas = draw.boxes(pieces_image, board.pieces)
    draw.save("pieces_colors", canvas)

    board.pieces = objects.process_pieces_amount(board.pieces)
    canvas = draw.boxes(pieces_image, board.pieces)
    draw.save("pieces_amount", canvas)

    canny = create_canny(board.image)
    board.corners = find_corners(canny)
    if board.corners is None:
        log.error("error finding corners of board")
        log.error(bad_picture_msg)
        return bad_picture_msg

    log.info(f"corners found: {board.corners}")
    canny_warped, warp_matrix_inverse = warp(canny, board.corners)

    vert, hori = lines.find_warped_lines(canny_warped)
    if vert is None or hori is None:
        log.error("error finding lines of warped board")
        log.error(bad_picture_msg)
        return bad_picture_msg

    inters = intersect.calculate_all(vert, hori)
    if (failed := inters.shape != (9, 9, 2)) or debug:
        canvas = draw.points(canny_warped, inters)
        draw.save("intersections", canvas)
        if failed:
            log.error("must have 81 intersections in 9 rows and 9 columns")
            log.error(f"{inters.shape=}")
            log.error(bad_picture_msg)
            return bad_picture_msg

    inters = translate_inters(inters, warp_matrix_inverse, translate_params)
    # if debug:
    canvas = draw.points(BGR, inters)
    draw.save("translated_intersections", canvas)
    inters = translate_pieces(inters, pieces_params)
    # if debug:
    canvas = draw.points(pieces_image, inters)
    draw.save("translated_intersections", canvas)

    board.squares = squares.calculate(inters)
    # if debug:
    canvas = draw.squares(pieces_image, board.squares)
    draw.save("squares", canvas)

    board.squares, pieces = squares.fill(board.squares, board.pieces)
    board.squares, changed = squares.check_colors(pieces_image, board.squares)
    # if debug and changed:
    canvas = draw.squares(pieces_image, board.squares)
    draw.save("squares_check_colors", canvas)

    board.fen = fen.generate(board.squares)
    fen.dump(board.fen)
    # shutil.move(filename, "good/")
    return board.fen


def crop_image(image, boardbox):
    log.info("cropping image to board box...")
    translate_params = SimpleNamespace()
    x0, y0, x1, y1 = boardbox
    cropped = image[y0:y1, x0:x1]

    log.info("reducing cropped image to default size...")
    resize_factor = WIDTH_BOARD / cropped.shape[1]
    height_board = round(resize_factor * cropped.shape[0])
    cropped = cv2.resize(cropped, (WIDTH_BOARD, height_board))

    translate_params.x0 = x0
    translate_params.y0 = y0
    translate_params.resize_factor = resize_factor

    if debug:
        draw.save("cropped", cropped)
    return cropped, translate_params


def crop_pieces(image, boardbox, pieces):
    log.info("cropping image to board box...")
    translate_params = SimpleNamespace()
    x0, y0_board, x1, y1 = boardbox
    y0_pieces = np.min(pieces[:, 1])
    y0 = min(y0_pieces, y0_board)
    cropped = image[y0:y1, x0:x1]

    log.info("reducing cropped image to default size...")
    resize_factor = WIDTH_BOARD / cropped.shape[1]
    height_board = round(resize_factor * cropped.shape[0])
    cropped = cv2.resize(cropped, (WIDTH_BOARD, height_board))

    translate_params.x0 = x0
    translate_params.y0 = y0
    translate_params.resize_factor = resize_factor

    if debug:
        draw.save("pieces_image", cropped)
    return cropped, translate_params


def create_canny(image):
    log.info("pre-processing image...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsvalue = hsv[:, :, 2]
    del hsv

    log.info("applying distributed histogram equalization to image...")
    clip_limit = 1.0
    grid = (10, 10)
    clahe = cv2.createCLAHE(clip_limit, grid)
    gray = clahe.apply(gray)
    hsvalue = clahe.apply(hsvalue)

    if debug:
        draw.save("clahe_gray", gray)
        draw.save("clahe_hsvalue", hsvalue)

    log.info("finding edges for gray and hsvalue images...")
    canny_gray = find_edges(gray)
    canny_hsvalue = find_edges(hsvalue)
    canny = cv2.bitwise_or(canny_gray, canny_hsvalue)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    canny = cv2.morphologyEx(canny, cv2.MORPH_DILATE, kernel)
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

    if debug:
        draw.save("canny_gray", canny_gray)
        draw.save("canny_hsvalue", canny_hsvalue)
        draw.save("canny", canny)
    return canny


def find_edges(image):
    log.info("filtering image...")
    f = np.array(image, dtype='float64')
    weights = np.empty(image.shape, dtype='float64')
    normalization = np.empty(image.shape, dtype='float64')
    g = np.empty(image.shape, dtype='float64')

    lfilter(f, g, weights, normalization, f.shape[0])
    lfilter(g, f, weights, normalization, f.shape[0])
    lfilter(f, g, weights, normalization, f.shape[0])

    g = np.round(g)
    g = np.clip(g, 0, 255)
    g = np.array(g, dtype='uint8')
    if debug:
        draw.save("lowpass", g)

    canny_mean_threshold = 9
    threshold_high0 = 250
    canny = find_canny(g, canny_mean_threshold, threshold_high0)
    return canny


def find_canny(image, canny_mean_threshold=8, threshold_high0=250):
    log.info("finding edges with Canny...")

    got_canny = False

    canny_threshold_high_min = 30
    canny_threshold_low_min = 10

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
        log.info(f"failed to find edges with"
                 f"mean >= {canny_mean_threshold:0=.1f}")
        log.info(f"last canny thresholds: {threshold_low, threshold_high}")
    return canny


def warp(canny, corners):
    log.debug("transforming perspective...")
    TL = corners[0]
    TR = corners[1]
    BR = corners[2]
    BL = corners[3]
    orig_points = np.array([TL, TR, BR, BL], dtype="float32")

    width = height = WIDTH_BOARD

    newshape = [[0, 0], [width, 0], [width, height], [0, height]]
    newshape = np.array(newshape, dtype='float32')
    warp_matrix = cv2.getPerspectiveTransform(orig_points, newshape)
    _, warp_matrix_inverse = cv2.invert(warp_matrix)
    canny_warped = cv2.warpPerspective(canny, warp_matrix, (width, height))

    if debug:
        draw.save("canny_warped", canny_warped)
    return canny_warped, warp_matrix_inverse


def translate_inters(inters, warp_matrix_inverse, translate_params):
    inters = np.array(inters, dtype='float64')

    inters = cv2.perspectiveTransform(inters, warp_matrix_inverse)
    inters[:, :, 0] /= translate_params.resize_factor
    inters[:, :, 1] /= translate_params.resize_factor

    inters[:, :, 0] += translate_params.x0
    inters[:, :, 1] += translate_params.y0

    return np.array(np.round(inters), dtype='int32')


def translate_pieces(inters, translate_params):
    inters = np.array(inters, dtype='float64')

    inters[:, :, 0] -= translate_params.x0
    inters[:, :, 1] -= translate_params.y0

    inters[:, :, 0] *= translate_params.resize_factor
    inters[:, :, 1] *= translate_params.resize_factor

    return np.array(np.round(inters), dtype='int32')


def find_corners(canny):
    vert, hori = lines.find_diagonal_lines(canny)
    if vert is None or hori is None:
        log.error("error finding diagonal lines")
        return None

    inters = intersect.calculate_all(vert, hori)
    if debug:
        canvas = draw.points(canny, inters)
        draw.save("warped_inters", canvas)

    log.debug("calculating 4 corners of board...")
    inters = inters.reshape((-1, 2))
    points_sum = np.zeros((inters.shape[0]), dtype='int32')
    points_sub = np.zeros((inters.shape[0]), dtype='int32')

    points_sum[:] = inters[:, 0] + inters[:, 1]
    points_sub[:] = inters[:, 0] - inters[:, 1]

    TL = inters[np.argmin(points_sum)]
    TR = inters[np.argmax(points_sub)]
    BR = inters[np.argmax(points_sum)]
    BL = inters[np.argmin(points_sub)]

    corners = np.array([TL, TR, BR, BL], dtype='int32')
    if debug:
        canvas = draw.corners(canny, corners)
        draw.save("corners", canvas)
    return corners


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        print(main(filename))
