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
    fen_split = str.split(fen, " ", -1)
    if len(fen_split) > 1:
        fen_meta = " " + str.join(" ", str.split(fen, " ", -1)[1:])
    else:
        fen_meta = ""
    fen_squares = fen_split[0]

    for length in range(8, 1, -1):
        fen_squares = str.replace(fen_squares, "1"*length, str(length))

    return fen_squares + fen_meta


def decompress(fen):
    log.info("decompressing FEN...")
    fen_split = str.split(fen, " ", -1)
    if len(fen_split) > 1:
        fen_meta = " " + str.join(" ", str.split(fen, " ", -1)[1:])
    else:
        fen_meta = ""
    fen_squares = fen_split[0]

    for length in range(8, 1, -1):
        fen_squares = str.replace(fen_squares, str(length), "1"*length)

    return fen_squares + fen_meta


def dump(fen):
    invert = "\033[1;7m"
    reset = '\033[0;m'
    print(f"  {invert} A B C D E F G H {reset}")

    fen = re.sub(r'^', " ", fen)
    fen = re.sub(r'/', "\n ", fen)
    fen = re.sub(r'$', "", fen)
    fen = re.sub(r'([KQRBNP])', r'\1 ', fen, flags=re.IGNORECASE)
    fen = re.sub(r'([0-9])', lambda x: '· ' * int(x[0]), fen)

    for i, line in enumerate(str.splitlines(fen)):
        row = 8-i
        print(f"{invert} {row}{reset}{line}{invert}{row} {reset}", sep='')

    print(f"  {invert} A B C D E F G H {reset}")
    return


def validate(fen):
    return False


if __name__ == "__main__":
    for fen in sys.argv[1:]:
        if not validate(fen):
            print(f"Invalid fen: {fen}")
            continue
        print(decompress(fen))
        print(compress(fen))
        dump(fen)
