import cv2
from pathlib import Path
import logging as log

from squares import calc_squares
from lines import find_lines
import yolo_wrap as yolo
import fen as fen
import auxiliar as aux
import constants as consts


class Image:
    def __init__(self, filename):
        self.filename = filename
        self.basename = Path(self.filename).stem


def algorithm(filename):
    img = Image(filename)

    img.BGR = cv2.imread(img.filename)

    img = yolo.detect_objects(img)
    img = crop_board(img)
    img = reduce_box(img)
    img = pre_process(img)

    img = find_lines(img)
    img = calc_squares(img)

    img.longfen, img.fen = fen.generate(img.squares)
    fen.dump(img.longfen)
    return img.fen


def crop_board(img):
    log.info("cropping image to board box...")
    x0, y0, x1, y1 = img.boardbox
    d = consts.margin
    img.x0, img.y0 = x0 - d, y0 - d
    img.x1, img.y1 = x1 + d, y1 + d

    img.board = img.BGR[img.y0:img.y1, img.x0:img.x1]
    if aux.debugging():
        aux.save(img, "board_box", img.board)
    return img


def reduce_box(img):
    log.info(f"reducing cropped image to default size ({consts.bwidth})...")
    img.bwidth = consts.bwidth
    img.bfact = img.bwidth / img.board.shape[1]
    img.bheigth = round(img.bfact * img.board.shape[0])

    img.board = cv2.resize(img.board, (img.bwidth, img.bheigth))
    if aux.debugging():
        aux.save(img, "board_reduce", img.board)
    return img


def pre_process(img):
    log.info("creating HSV representation of image...")
    img.HSV = cv2.cvtColor(img.board, cv2.COLOR_BGR2HSV)
    img.V = img.HSV[:, :, 2]

    log.info("converting image to grayscale...")
    img.gray = cv2.cvtColor(img.board, cv2.COLOR_BGR2GRAY)
    if aux.debugging():
        aux.save(img, "gray_board", img.gray)
    log.info("generating 3 channel gray image for drawings...")
    img.gray3ch = cv2.cvtColor(img.gray, cv2.COLOR_GRAY2BGR)

    log.info("applying distributed histogram equalization to image...")
    tgs = consts.tileGridSize
    cliplim = consts.clipLimit
    clahe = cv2.createCLAHE(clipLimit=cliplim, tileGridSize=(tgs, tgs))
    img.G = clahe.apply(img.gray)
    img.V = clahe.apply(img.V)
    if aux.debugging():
        aux.save(img, "claheG", img.G)
        aux.save(img, "claheV", img.V)

    return img
