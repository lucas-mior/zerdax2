import sys
import cv2
import numpy as np
import logging as log
from types import SimpleNamespace

from squares import calc_squares
from lines import find_lines, calc_intersections
import yolo_wrap as yolo
import fen as fen
from filter import filter
import constants as consts
import drawings as draw

img = None
debug = False


def algorithm(filename):
    global img, debug
    debug = log.root.level < 20
    img = SimpleNamespace(filename=filename)
    img.basename = filename.rsplit(".", 1)[0]
    img.basename = img.basename.rsplit("/", 1)[-1]

    img.BGR = cv2.imread(img.filename)

    img = yolo.detect_objects(img)
    img = crop_board_to_size(img)
    img = pre_process(img)
    canny = create_cannys(img)

    vert, hori = find_lines(canny)

    if (lv := len(vert)) != 9 or (lh := len(hori)) != 9 or debug:
        canvas = draw.lines(img.gray3ch, vert, hori)
        draw.save("find_lines", canvas)
        if lv != 9 or lh != 9:
            log.error("There should be 9 vertical lines and",
                      "9 horizontal lines")
            log.error(f"Got {lv} vertical and {lh} horizontal lines")
            exit()

    inters = calc_intersections(vert, hori)
    if (failed := inters.shape != (9, 9, 2)) or debug:
        canvas = draw.points(img.gray3ch, inters)
        draw.save("intersections", canvas)
        if failed:
            log.error("There should be 81 intersections",
                      "in 9 rows and 9 columns")
            log.error(f"{inters.shape=}")
            exit()
    inters = np.array(inters, dtype='float64')
    # scale to input size
    inters[:, :, 0] /= img.bfact
    inters[:, :, 1] /= img.bfact
    # position board bounding box
    inters[:, :, 0] += img.x0
    inters[:, :, 1] += img.y0
    inters = np.array(np.round(inters), dtype='int32')
    if debug:
        canvas = draw.points(img.BGR, inters)
        draw.save("intersections", canvas)

    img = calc_squares(img, inters)

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

    log.info(f"reducing cropped image to default size ({consts.bwidth})...")
    img.bwidth = consts.bwidth
    img.bfact = img.bwidth / img.board.shape[1]
    img.bheigth = round(img.bfact * img.board.shape[0])

    img.board = cv2.resize(img.board, (img.bwidth, img.bheigth))
    if debug:
        draw.save("board", img.board)
    return img


def pre_process(img):
    log.info("creating HSV representation of image...")
    img.HSV = cv2.cvtColor(img.board, cv2.COLOR_BGR2HSV)
    img.V = img.HSV[:, :, 2]

    log.info("converting image to grayscale...")
    img.gray = cv2.cvtColor(img.board, cv2.COLOR_BGR2GRAY)
    if debug:
        draw.save("gray_board", img.gray)
    log.info("generating 3 channel gray image for drawings...")
    img.gray3ch = cv2.cvtColor(img.gray, cv2.COLOR_GRAY2BGR)

    log.info("applying distributed histogram equalization to image...")
    tgs = consts.tileGridSize
    cliplim = consts.clipLimit
    clahe = cv2.createCLAHE(clipLimit=cliplim, tileGridSize=(tgs, tgs))
    img.G = clahe.apply(img.gray)
    img.V = clahe.apply(img.V)
    if debug:
        draw.save("claheG", img.G)
        draw.save("claheV", img.V)

    return img


def create_cannys(img):
    log.info("finding edges for gray, and V images...")
    cannyG, got_cannyG = find_edges(img.G, lowpass=filter)
    cannyV, got_cannyV = find_edges(img.V, lowpass=filter)
    if not got_cannyG or not got_cannyV or debug:
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


def find_edges(image, lowpass):
    log.info("filtering image...")
    image = lowpass(image)
    if lowpass == filter:
        wmin = consts.wminfilter
        thigh0 = consts.thighfilter
    elif lowpass == gauss:
        wmin = consts.wmingauss
        thigh0 = consts.thighgauss
    canny, got_canny = find_canny(image, wmin, thigh0)
    if not got_canny or debug:
        draw.save("lowpass", image)
    return canny, got_canny


def find_canny(image, wmin=8, thigh0=250):
    log.info(f"finding edges with Canny until mean >= {wmin:0=.1f}...")

    got_canny = False
    thighmin = consts.thighmin
    tlowmin = consts.tlowmin
    thigh = thigh0
    while thigh >= thighmin:
        tlow = max(tlowmin, round(thigh*0.8))
        while tlow >= tlowmin:
            canny = cv2.Canny(image, tlow, thigh)
            w = np.mean(canny)
            if w >= wmin:
                log.info(f"{w:0=.2f} >= {wmin:0=.1f}, @ {tlow}, {thigh}")
                got_canny = True
                break
            else:
                log.debug(f"{w:0=.2f} < {wmin:0=.1f}, @ {tlow}, {thigh}")
                gain = wmin - w
                diff = round(max(8, gain*8))
                if tlow <= tlowmin:
                    break
                tlow = max(tlowmin, tlow - diff)

        if got_canny or (thigh <= thighmin):
            break
        else:
            diff = round(max(6, gain*(thigh/18)))
            thigh = max(thighmin, thigh - diff)

    if not got_canny:
        log.info(f"Failed to find edges with mean >= {wmin:0=.1f}")
        log.info(f"Last canny thresholds: {tlow, thigh}")

    return canny, got_canny


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        print(algorithm(filename))
