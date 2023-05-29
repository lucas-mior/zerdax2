import sys
import re
import logging as log

from misc import SYMBOLS


def generate(squares):
    long_fen = ""
    for i in range(7, -1, -1):
        for j in range(0, 8):
            sq = squares[j, i]
            long_fen += SYMBOLS[sq[4, 1]]
        long_fen += '/'
    long_fen = long_fen[:-1]
    log.info(f"{long_fen=}")
    return compress(long_fen)


def compress(fen):
    log.info("compressing FEN...")
    for length in range(8, 1, -1):
        fen = str.replace(fen, "1"*length, str(length))
    return fen


def decode(fen):
    log.info("decoding FEN...")
    for length in range(8, 1, -1):
        fen = str.replace(fen, str(length), "1"*length)
    return fen


def dump(fen):
    invert = "\033[1;7m"
    reset = '\033[0;m'
    print(f"  {invert} A B C D E F G H {reset}")

    fen = re.sub(r'^', " ", fen)
    fen = re.sub(r'/', "\n ", fen)
    fen = re.sub(r'$', "", fen)
    fen = re.sub(r'([a-zA-Z])', r'\1 ', fen)
    fen = re.sub(r'([0-9])', lambda x: '· ' * int(x[0]), fen)
    for i, line in enumerate(fen.splitlines()):
        row = 8-i
        print(f"{invert} {row}{reset}{line}{invert}{row} {reset}", sep='')

    print(f"  {invert} A B C D E F G H {reset}")
    return


if __name__ == "__main__":
    for fen in sys.argv[1:]:
        print(decode(fen))
        print(compress(fen))
        dump(fen)
