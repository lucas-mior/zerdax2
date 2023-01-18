import cv2
import numpy as np
from pathlib import Path
import logging as log

from find_squares import find_squares
import yolo_wrap as yolo
import fen as fen
import auxiliar as aux
import constants as consts
import drawings as draw


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

    img = find_squares(img)
    img.squares = fill_squares(img.squares, img.pieces)
    img.squares = check_bottom_right(img, img.BGR, img.squares)
    if aux.debugging():
        canvas = draw.squares(img.BGR, img.squares)
        aux.save(img, "A1E4C5H8", canvas)

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
    # img.H = img.HSV[:, :, 0]
    # img.S = img.HSV[:, :, 1]
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


def fill_squares(squares, pieces):
    log.info("filling squares...")
    piece_y_tol = consts.piece_y_tol
    for i in range(7, -1, -1):
        for j in range(0, 8):
            sq = squares[j, i]
            got_piece = False
            for piece in pieces:
                x0, y0, x1, y1, _, number = piece[:6]
                xm = round((x0 + x1)/2)
                y = round(y1) - piece_y_tol
                if cv2.pointPolygonTest(sq, (xm, y), True) >= 0:
                    sq[4] = [1, number]
                    got_piece = True
                    pieces.remove(piece)
                    break
            if not got_piece:
                sq[4] = [0, -1]

    return squares


def check_bottom_right(img, image, squares):
    a8 = np.copy(squares[7, 0])
    contour = a8[:4]
    frame = cv2.boundingRect(contour)
    x0, y0, dx, dy = frame
    contour[:, 0] -= x0
    contour[:, 1] -= y0
    a = image[y0:y0+dy, x0:x0+dx]
    b = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    mask1 = np.zeros(b.shape, dtype='uint8')
    cv2.drawContours(mask1, [contour], -1, 255, -1)
    mask0 = cv2.bitwise_not(mask1)
    mean0 = cv2.mean(b, mask=mask0)[0]
    mean1 = cv2.mean(b, mask=mask1)[0]
    if a8[4, 1] < 0:
        pass
    elif a8[4, 1] <= 6:
        mean1 -= 30
    else:
        mean1 += 30
    mean0, mean1 = round(mean0), round(mean1)
    if mean1 < mean0:
        if squares[0, 0, 0, 1] > squares[1, 0, 0, 1]:
            squares = np.rot90(squares, k=1)
        else:
            squares = np.rot90(squares, k=3)
    return squares
