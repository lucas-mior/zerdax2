import cv2
from pathlib import Path
import re

from find_board import find_board
from find_squares import find_squares
from find_pieces import detect_objects
from generate_fen import generate_fen
import auxiliar as aux

# WIDTH = 1280
BWIDTH = 640


class Image:
    def __init__(self, filename):
        self.filename = filename
        self.basename = Path(self.filename).stem


def crop_board(img):
    print("cropping image to board box...")
    b = img.boardbox
    x0, y0 = int(b[0]), int(b[1])
    x1, y1 = int(b[2]), int(b[3])
    img.x0, img.y0 = x0 - 10, y0 - 10
    img.x1, img.y1 = x1 + 10, y1 + 10

    img.board = img.BGR[img.y0:img.y1, img.x0:img.x1]
    aux.save(img, "board_box", img.board)

    return img


def pre_process(img):
    print("creating HSV representation of image...")
    img.HSV = cv2.cvtColor(img.board, cv2.COLOR_BGR2HSV)
    img.H = img.HSV[:, :, 0]
    img.S = img.HSV[:, :, 1]
    img.V = img.HSV[:, :, 2]

    print("converting image to grayscale...")
    img.gray = cv2.cvtColor(img.board, cv2.COLOR_BGR2GRAY)
    aux.save(img, "gray_board", img.gray)

    print("applying gaussian blur...")
    img.G = cv2.GaussianBlur(img.gray, (7, 7), 0.3)
    img.V = cv2.GaussianBlur(img.V, (7, 7), 0.3)
    aux.save(img, "Gblur", img.G)
    aux.save(img, "Vblur", img.V)
    # img.G = lf.ffilter(img.gray)
    # img.V = lf.ffilter(img.V)

    print("applying distributed histogram equalization to image...")
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
    img.claheG = clahe.apply(img.G)
    img.claheV = clahe.apply(img.V)
    aux.save(img, "Gclahe", img.claheG)
    aux.save(img, "Vclahe", img.claheV)

    print("generating 3 channel gray image for drawings...")
    img.gray3ch = cv2.cvtColor(img.gray, cv2.COLOR_GRAY2BGR)
    return img


def algorithm(filename, log):
    img = Image(filename)
    img.log = log

    img.BGR = cv2.imread(img.filename)

    img = detect_objects(img)
    img = crop_board(img)
    img = reduce_box(img)
    img = pre_process(img)

    img = find_board(img)
    img = find_squares(img)
    aux.save(img, "yolo", img.yolopieces)
    img = generate_fen(img)

    aux.draw_fen_terminal(img.longfen)

    return img.fen


def reduce_box(img):
    print("reduce cropped image to default size...")
    img.bwidth = BWIDTH
    img.bfact = img.bwidth / img.board.shape[1]
    img.bheigth = round(img.bfact * img.board.shape[0])

    img.board = cv2.resize(img.board, (img.bwidth, img.bheigth))
    aux.save(img, "board_reduce", img.board)
    return img
