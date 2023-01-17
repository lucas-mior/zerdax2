import cv2
import re
import logging as log
from zerdax2_misc import SYMBOLS


def generate(squares, pieces):
    longfen = create(squares, pieces)
    log.info(f"{longfen=}")
    fen = compress(longfen)
    return longfen, fen


def create(squares, pieces):
    log.info("creating fen...")
    fen = ""
    for i in range(7, -1, -1):
        for j in range(0, 8):
            sq = squares[j, i]
            got_piece = False
            for piece in pieces:
                x0, y0, x1, y1, _, number = piece[:6]
                xm = round((x0 + x1)/2)
                y = round(y1) - 15
                if cv2.pointPolygonTest(sq, (xm, y), True) >= 0:
                    fen += SYMBOLS[int(number)]
                    got_piece = True
                    pieces.remove(piece)
                    break
            if not got_piece:
                fen += '1'
        fen += '/'

    return fen[:-1]


def compress(fen):
    log.info("compressing FEN...")
    for length in reversed(range(2, 9)):
        fen = fen.replace(length * '1', str(length))

    return fen


def dump(fen):
    print("―"*19)

    print("| ", end='')
    fen = re.sub(r'/', "|\n| ", fen)
    fen = re.sub(r'([a-zA-Z])', r'\1 ', fen)
    fen = re.sub(r'(1)', r'· ', fen)
    print(fen, end='')
    print("|")

    print("―"*19)
    return
