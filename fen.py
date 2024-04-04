import sys
import re
import logging as log

from misc import SYMBOLS


def generate(squares):
    long_fen = ""
    for i in range(7, -1, -1):
        for j in range(0, 8):
            sq = squares[i, j]
            long_fen += SYMBOLS[sq[4, 1]]
        long_fen += '/'
    long_fen = long_fen[:-1]
    log.info(f"{long_fen=}")
    return compress(long_fen)


def split(fen):
    fen_split = str.split(fen, " ", -1)
    if len(fen_split) > 1:
        fen_meta = " " + str.join(" ", str.split(fen, " ", -1)[1:])
    else:
        fen_meta = ""
    fen_squares = fen_split[0]
    return fen_squares, fen_meta


def compress(fen):
    log.info("compressing FEN...")
    fen_squares, fen_meta = split(fen)

    for length in range(8, 1, -1):
        fen_squares = str.replace(fen_squares, "1"*length, str(length))

    return fen_squares + fen_meta


def decompress(fen):
    log.info("decompressing FEN...")
    fen_squares, fen_meta = split(fen)

    for length in range(8, 1, -1):
        fen_squares = str.replace(fen_squares, str(length), "1"*length)

    return fen_squares + fen_meta


def dump(fen):
    fblack = "\033[01;38;2;000;000;000m"
    fgray = "\033[01;38;2;200;200;200m"
    fwhite = "\033[01;38;2;255;255;255m"

    bblack = "\033[01;48;2;000;000;000m"
    bgray = "\033[01;48;2;200;200;200m"
    bwhite = "\033[01;48;2;255;255;255m"
    reset = '\033[0;m'

    print(f"  {bwhite}{fblack} A B C D E F G H {reset} ", sep="")

    fen = re.sub(r'^', " ", fen)
    fen = re.sub(r'/', "\n ", fen)
    fen = re.sub(r'$', "", fen)
    fen = re.sub(r'([KQRBNP])', r'\1 ', fen, flags=re.IGNORECASE)
    fen = re.sub(r'([0-9])', lambda x: '· ' * int(x[0]), fen)

    for i, line in enumerate(str.splitlines(fen)):
        row = 8 - i
        print(bblack, fwhite, sep="", end="")
        row1 = f"{bwhite}{fblack} {row}"
        row2 = f"{bwhite}{fblack}{row}"
        line = f"{bblack}{fwhite}{line}"
        print(f"{row1}{line}{row2} {reset}", sep='')

    print(f"  {bwhite}{fblack} A B C D E F G H {reset}")
    return


def validate(fen):
    return True


def compressed_align(fen):
    rows = str.split(fen, "/")
    new = ""
    for row in rows:
        left = (8 - len(row))*" "
        new += f"{row}{left}/"
    new = new[:-1]
    return new


if __name__ == "__main__":
    for fen in sys.argv[1:]:
        if not validate(fen):
            print(f"invalid fen: {fen}")
            continue
        long_fen = decompress(fen)
        compressed_fen = compress(long_fen)
        compressed_fen_align = compressed_align(fen)

        print(long_fen)
        print(compressed_fen_align)
        print(compressed_fen)

        dump(fen)
