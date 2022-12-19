import cv2
from pathlib import Path
import re

from find_board import find_board
from find_squares import find_squares
from find_pieces import find_pieces
from generate_fen import generate_fen
import auxiliar as aux


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


def algorithm(filename, log):
    img = Image(filename)
    img.log = log
    img = read_image(img)

    img = find_board(img)
    img = find_squares(img)
    img = find_pieces(img)
    # aux.save(img, "yolo", img.yolopieces)
    img = generate_fen(img)

    draw_fen_terminal(img.longfen)

    return img.fen


def draw_fen_terminal(fen):
    print("―"*19)

    print("| ", end='')
    fen = re.sub(r'/', "|\n| ", fen)
    fen = re.sub(r'([a-zA-Z])', r'\1 ', fen)
    fen = re.sub(r'(1)', r'· ', fen)
    print(fen, end='')
    print("|")

    print("―"*19)


def read_image(img):
    print("reading image in BGR...")
    img.BGR = cv2.imread(img.filename)
    print("reducing image to 1000 width...")
    img = reduce(img)
    print("creating HSV representation of image...")
    img.HSV = cv2.cvtColor(img.BGR, cv2.COLOR_BGR2HSV)
    print("split HSV into channels...")
    img.H = img.HSV[:, :, 0]
    img.S = img.HSV[:, :, 1]
    img.V = img.HSV[:, :, 2]

    print("converting image to grayscale...")
    img.gray = cv2.cvtColor(img.BGR, cv2.COLOR_BGR2GRAY)
    print("generating 3 channel gray image for drawings...")
    img.gray3ch = cv2.cvtColor(img.gray, cv2.COLOR_GRAY2BGR)

    return img
