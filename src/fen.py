import re
import logging as log
from zerdax2_misc import SYMBOLS


def generate(squares):
    longfen = create(squares)
    log.info(f"{longfen=}")
    fen = compress(longfen)
    return longfen, fen


def create(squares):
    fen = ""
    for i in range(7, -1, -1):
        for j in range(0, 8):
            sq = squares[j, i]
            fen += SYMBOLS[sq[4, 1]]
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
