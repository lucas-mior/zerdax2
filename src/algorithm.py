import cv2
from pathlib import Path
import re

from find_board import find_board
from find_squares import find_squares
from find_pieces import detect_objects
from generate_fen import generate_fen
import auxiliar as aux

WIDTH = 1280
BOARD_WIDTH = 960


class Image:
    def __init__(self, filename):
        self.filename = filename
        self.basename = Path(self.filename).stem


def reduce(img):
    img.width = 1000
    img.fact = img.width / img.BGR.shape[1]
    img.heigth = round(img.fact * img.BGR.shape[0])

    img.BGR = cv2.resize(img.BGR, (img.width, img.heigth))
    img.area = img.heigth * img.width
    return img


def crop_board(img):
    print("cropping image to board box...")
    b = img.board
    x0, y0 = int(b[0]), int(b[1])
    x1, y1 = int(b[2]), int(b[3])
    img.BGR = img.BGR[y0:y1, x0:x1]
    aux.save(img, "board_box", img.BGR)

    return img


def pre_process(img):
    print("creating HSV representation of image...")
    img.HSV = cv2.cvtColor(img.BGR, cv2.COLOR_BGR2HSV)
    img.H = img.HSV[:, :, 0]
    img.S = img.HSV[:, :, 1]
    img.V = img.HSV[:, :, 2]

    print("converting image to grayscale...")
    img.gray = cv2.cvtColor(img.BGR, cv2.COLOR_BGR2GRAY)
    print("generating 3 channel gray image for drawings...")
    img.gray3ch = cv2.cvtColor(img.gray, cv2.COLOR_GRAY2BGR)
    return


def algorithm(filename, log):
    img = Image(filename)
    img.log = log

    img.BGR = cv2.imread(img.filename)

    img = detect_objects(img)
    img = crop_board(img)
    img = pre_process(img)
    img = reduce_box(img)

    img = find_board(img)
    img = find_squares(img)
    aux.save(img, "yolo", img.yolopieces)
    img = generate_fen(img)

    aux.draw_fen_terminal(img.longfen)

    return img.fen


def reduce_box(img):
    print("reducing images to default size...")
    img.hwidth = BOARD_WIDTH
    img.hfact = img.hwidth / img.gray.shape[1]
    img.hheigth = round(img.hfact * img.gray.shape[0])
    img.harea = img.hwidth * img.hheigth
    nsh = (img.hwidth, img.hheigth)
    innsh = (img.hwidth - 10, img.hheigth - 10)

    print(f"reducing all images to {img.hwidth} width")
    img.G = cv2.resize(img.G, nsh)
    img.V = cv2.resize(img.V, nsh)
    img.claheG = cv2.resize(img.claheG, nsh)
    img.claheV = cv2.resize(img.claheV, nsh)
    img.dcont = cv2.resize(img.dcont, nsh)
    img.fedges = cv2.resize(img.fedges, nsh)
    img.gray = cv2.resize(img.gray, nsh)
    img.BGR = cv2.resize(img.BGR, nsh)
    img.BGR_name = f"{img.basename}BGR.png"
    cv2.imwrite(img.BGR_name, img.BGR)
    img.gray3ch = cv2.resize(img.gray3ch, nsh)
    img.inside = cv2.resize(img.inside, innsh)

    return img
