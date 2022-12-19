import cv2
import numpy as np
from zerdax2 import COLORS, SYMBOLS, NAMES, PIECES, CLASSES


def generate_fen(img):
    img = create_fen(img)
    img.fen = compress_fen(img.longfen)
    return img


def create_fen(img):
    print("creating fen...")
    fen = ''
    for i in range(7, -1, -1):
        for j in range(0, 8):
            sq = img.sqback[j, i]
            got_piece = False
            for piece in img.pieces:
                xm = round((piece[2] + piece[0])/2)
                y = round(piece[3]) - 15
                p = (xm, y)
                if cv2.pointPolygonTest(sq, p, True) >= 0:
                    fen += SYMBOLS[str(int(piece[5]))]
                    got_piece = True
                    img.pieces.remove(piece)
                    break
            if not got_piece:
                fen += '1'
        fen += '/'
    fen = fen[:-1]

    img.longfen = fen
    print("long fen:", img.longfen)
    return img


def compress_fen(fen):
    print("generating compressed FEN...")
    for length in reversed(range(2, 9)):
        fen = fen.replace(length * '1', str(length))

    return fen
