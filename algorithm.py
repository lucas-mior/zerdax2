import sys
import cv2
import numpy as np
import logging as log
from types import SimpleNamespace

import algorithm as algo
import squares as squares
import lines as lines
import intersections as intersections
import perspective as perspective
import yolo_wrap as yolo
import fen as fen
import constants as consts
import drawings as draw
from c_load import lfilter

img = None
debug = False
bad_picture_message = ""


def algorithm(filename):
    global img, debug, bad_picture_message
    debug = log.root.level < 20
    img = SimpleNamespace(filename=filename)
    img.basename = filename.rsplit(".", 1)[0]
    img.basename = img.basename.rsplit("/", 1)[-1]
    bad_picture_message = f"{img.filename}: bad picture, try again from another angle"

    img.BGR = cv2.imread(img.filename)

    img.boardbox, img.pieces = yolo.detect_objects(img.filename)

    img.pieces = yolo.determine_colors(img.pieces, img.BGR)
    img.pieces = yolo.process_pieces(img.pieces)

    if (failed := img.boardbox is None) or algo.debug:
        canvas = draw.boxes(img.BGR, img.pieces)
        draw.save("yolo", canvas)
        if failed:
            log.error(bad_picture_message)
            return bad_picture_message

    log.info(f"Board detected: {img.boardbox}")
    img = crop_board_to_size(img)
    if img.board.shape[0] < 300:
        log.error(bad_picture_message)
        return bad_picture_message

    img = pre_process(img)
    canny = create_cannys(img)
    if algo.debug:
        draw.save("edges", canny)

    img.corners = lines.find_corners(canny)
    log.info(f"Corners found: {img.corners}")
    if algo.debug:
        canvas = draw.corners(img.board, img.corners)
        draw.save("corners", canvas)
    canny_warped, warp_inverse_matrix = perspective.transform(canny, img.corners)
    warp3ch = cv2.cvtColor(canny_warped, cv2.COLOR_GRAY2BGR)

    vert, hori = lines.find_wlines(canny_warped)
    if vert is None or hori is None:
        log.error(bad_picture_message)
        return bad_picture_message

    lv, lh = len(vert), len(hori)
    if (failed := (lv != 9 or lh != 9)) or algo.debug:
        canvas = draw.lines(warp3ch, vert, hori)
        draw.save("find_lines", canvas)
        if failed:
            log.error("There should be 9 vertical lines and",
                      "9 horizontal lines")
            log.error(f"Got {lv} vertical and {lh} horizontal lines")
            log.error(bad_picture_message)
            return bad_picture_message

    inters = intersections.calculate_all(vert, hori)
    if (failed := inters.shape != (9, 9, 2)) or algo.debug:
        canvas = draw.points(warp3ch, inters)
        draw.save("intersections", canvas)
        if failed:
            log.error("There should be 81 intersections",
                      "in 9 rows and 9 columns")
            log.error(f"{inters.shape=}")
            log.error(bad_picture_message)
            return bad_picture_message

    inters = np.array(inters, dtype='float64')

    inters = cv2.perspectiveTransform(inters, warp_inverse_matrix)
    # scale to input size
    inters[:, :, 0] /= img.resize_factor
    inters[:, :, 1] /= img.resize_factor
    # position board bounding box
    inters[:, :, 0] += img.x0
    inters[:, :, 1] += img.y0
    inters = np.array(np.round(inters), dtype='int32')
    if algo.debug:
        canvas = draw.points(img.BGR, inters)
        draw.save("intersections", canvas)

    img.squares = squares.calculate(inters)
    if algo.debug:
        canvas = draw.squares(img.BGR, img.squares)
        draw.save("A1E4C5H8", canvas)

    img.squares, pieces = squares.fill(img.squares, img.pieces)
    if len(pieces) > 0:
        img.squares, pieces = squares.fill(img.squares, img.pieces, force=True)

    img.squares, changed = squares.check_colors(img.BGR, img.squares)
    if algo.debug and changed:
        canvas = draw.squares(img.BGR, img.squares)
        draw.save("A1E4C5H8", canvas)

    img.fen = fen.generate(img.squares)
    fen.dump(img.fen)
    return img.fen


def crop_board_to_size(img):
    log.info("cropping image to board box...")
    x0, y0, x1, y1 = img.boardbox
    d = consts.margin
    img.x0, img.y0 = x0 - d, y0 - d
    img.x1, img.y1 = x1 + d, y1 + d
    img.board = img.BGR[img.y0:img.y1, img.x0:img.x1]

    log.info(f"reducing cropped image to default size ({consts.width_board})...")
    img.width_board = consts.width_board
    img.resize_factor = img.width_board / img.board.shape[1]
    img.height_board = round(img.resize_factor * img.board.shape[0])

    img.board = cv2.resize(img.board, (img.width_board, img.height_board))
    if algo.debug:
        draw.save("board", img.board)
    return img


def pre_process(img):
    log.info("creating HSV representation of image...")
    img.HSV = cv2.cvtColor(img.board, cv2.COLOR_BGR2HSV)
    img.V = img.HSV[:, :, 2]

    log.info("converting image to grayscale...")
    img.gray = cv2.cvtColor(img.board, cv2.COLOR_BGR2GRAY)
    if algo.debug:
        draw.save("gray_board", img.gray)
    log.info("generating 3 channel gray image for drawings...")
    img.gray3ch = cv2.cvtColor(img.gray, cv2.COLOR_GRAY2BGR)

    log.info("applying distributed histogram equalization to image...")
    grid = (consts.tile_grid_size, consts.tile_grid_size)
    clip_limit = consts.clip_limit
    clahe = cv2.createCLAHE(clip_limit, grid)
    img.G = clahe.apply(img.gray)
    img.V = clahe.apply(img.V)
    if algo.debug:
        draw.save("claheG", img.G)
        draw.save("claheV", img.V)

    return img


def create_cannys(img):
    log.info("finding edges for gray, and V images...")
    cannyG, got_cannyG = find_edges(img.G, lowpass=filter)
    cannyV, got_cannyV = find_edges(img.V, lowpass=filter)
    if not got_cannyG or not got_cannyV or algo.debug:
        draw.save("cannyG", cannyG)
        draw.save("cannyV", cannyV)
    canny = cv2.bitwise_or(cannyG, cannyV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    canny = cv2.morphologyEx(canny, cv2.MORPH_DILATE, kernel)
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    return canny


def gauss(image):
    ks = consts.gauss_kernel_shape
    gamma = consts.gauss_gamma
    filtered = cv2.GaussianBlur(image, ks, gamma)
    return filtered


def filter(image):
    f = np.array(image, dtype='float64')
    W = np.zeros(image.shape, dtype='float64')
    N = np.zeros(image.shape, dtype='float64')
    g = np.zeros(image.shape, dtype='float64')

    lfilter(f, f.shape[0], f.shape[1], W, N, g, 1)
    lfilter(g, f.shape[0], f.shape[1], W, N, f, 1)
    lfilter(f, f.shape[0], f.shape[1], W, N, g, 1)

    g = np.round(g)
    g = np.clip(g, 0, 255)
    return np.array(g, dtype='uint8')


def find_edges(image, lowpass):
    log.info("filtering image...")
    image = lowpass(image)
    if lowpass == filter:
        canny_mean_threshold = consts.canny_mean_threshold_filter
        threshold_high0 = consts.threshold_highfilter
    elif lowpass == gauss:
        canny_mean_threshold = consts.canny_mean_threshold_gauss
        threshold_high0 = consts.threshold_highgauss
    canny, got_canny = find_canny(image, canny_mean_threshold, threshold_high0)
    if not got_canny or algo.debug:
        draw.save("lowpass", image)
    return canny, got_canny


def find_canny(image, canny_mean_threshold=8, threshold_high0=250):
    log.info(f"finding edges with Canny until mean >= {canny_mean_threshold:0=.1f}...")

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
        else:
            diff = round(max(6, gain*(threshold_high/18)))
            threshold_high = max(canny_threshold_high_min, threshold_high - diff)

    if not got_canny:
        log.info(f"Failed to find edges with mean >= {canny_mean_threshold:0=.1f}")
        log.info(f"Last canny thresholds: {threshold_low, threshold_high}")

    return canny, got_canny


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        print(algorithm(filename))
