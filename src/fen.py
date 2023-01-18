import sys
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


def dump(longfen):
    print("―"*19)

    print("| ", end='')
    longfen = re.sub(r'/', "|\n| ", longfen)
    longfen = re.sub(r'([a-zA-Z])', r'\1 ', longfen)
    longfen = re.sub(r'(1)', r'· ', longfen)
    print(longfen, end='')
    print("|")

    print("―"*19)
    return


if __name__ == "__main__":
    for longfen in sys.argv[1:]:
        print(compress(longfen))
        dump(longfen)
