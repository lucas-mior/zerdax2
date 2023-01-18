import cv2
import re
import logging as log
from zerdax2_misc import SYMBOLS
import constants as consts

piece_y_tol = consts.piece_y_tol


def generate(squares, pieces):
    longfen = create(squares, pieces)
    log.info(f"{longfen=}")
    fen = compress(longfen)
    return longfen, fen


def create(squares, pieces):
    fen = 8*'11111111/'
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
