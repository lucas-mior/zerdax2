import re
import logging as log
from zerdax2_misc import SYMBOLS


def generate(squares):
    longfen = ""
    for i in range(7, -1, -1):
        for j in range(0, 8):
            sq = squares[j, i]
            longfen += SYMBOLS[sq[4, 1]]
        longfen += '/'
    longfen = longfen[:-1]
    log.info(f"{longfen=}")
    fen = compress(longfen)
    return longfen, fen


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
